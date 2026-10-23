"""Stability measures; accuracy is reported only when explicit truth is present."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .geometry import palm_scale, center
from .recording import load_frames, json_safe


def finite_points(value):
    if value is None:
        return None
    x=np.asarray(value,dtype=float)
    return x if x.shape==(21,3) and np.isfinite(x).all() else None


def lag_estimate(times,raw,filtered):
    """Relative wrist-x lag on ONE contiguous, sufficiently varying segment."""
    if len(times)<30:
        return None
    t=np.asarray(times); x=np.asarray(raw); y=np.asarray(filtered)
    if np.std(x)<.005 or np.any(np.diff(t)<=0):
        return None
    dt=float(np.median(np.diff(t)))
    if np.max(np.diff(t))>3*dt:
        return None
    grid=np.arange(t[0],t[-1],dt)
    a=np.interp(grid,t,x); b=np.interp(grid,t,y)
    max_shift=min(int(.25/dt),len(grid)//4)
    if max_shift<1:
        return None
    correlations=[]
    for shift in range(max_shift+1):
        left=a[:len(a)-shift or None]; right=b[shift:]
        correlations.append(float(np.corrcoef(left,right)[0,1]) if min(np.std(left),np.std(right))>1e-9 else -1)
    best=int(np.argmax(correlations))
    if correlations[best]<.6:
        return None
    return {'milliseconds':best*dt*1000,'correlation':correlations[best],
            'at_search_boundary':best==max_shift,'resolution_ms':dt*1000}


def summarize(frames,jump_threshold=1.):
    frames=list(frames)
    if not frames:
        raise ValueError('분석할 프레임이 없습니다.')
    grouped=defaultdict(list)
    for f in frames:
        grouped[(f['scenario'],f['segment'])].append(f)
    summaries=[]
    for (scenario,segment),rows in grouped.items():
        for name in rows[0]['outputs']:
            valid=[]; steps=[]; speeds=[]; reappear=[]; recovery=[]; errors=[]
            runs=[]; run=[]; prev=None; prev_t=None; last_visible=None
            previous_raw_missing=False; recovery_start=None; unfinished=0
            predicted=0; resync=0; processing=[]
            for f in rows:
                o=f['outputs'][name]; x=finite_points(o['points']); raw=finite_points(f['raw']); t=f['t']
                processing.append(o['processing_ms']); predicted+=bool(o['predicted']); resync+=bool(o['resynced'])
                if raw is not None and previous_raw_missing:
                    if x is not None and last_visible is not None:
                        reappear.append(float(np.max(np.linalg.norm(x-last_visible,axis=1))/palm_scale(raw)))
                previous_raw_missing=raw is None
                if x is not None:
                    valid.append(x)
                    if prev is not None and t>prev_t:
                        step=float(np.max(np.linalg.norm(x-prev,axis=1))/palm_scale(prev))
                        steps.append(step); speeds.append(step/(t-prev_t))
                    truth=finite_points(f.get('truth'))
                    if truth is not None:
                        errors.append(float(np.mean((x-truth)**2)))
                    last_visible=x
                if x is not None and raw is not None and o['state']!='LOST':
                    run.append((t,float(raw[0,0]),float(x[0,0])))
                else:
                    if run: runs.append(run); run=[]
                if o['state']=='RECOVERING' and recovery_start is None:
                    recovery_start=t
                elif recovery_start is not None and o['state']=='NORMAL':
                    recovery.append(t-recovery_start); recovery_start=None
                elif recovery_start is not None and o['state'] in ('LOST','SUSPECT'):
                    unfinished+=1; recovery_start=None
                prev=x; prev_t=t
            if run: runs.append(run)
            if recovery_start is not None: unfinished+=1
            longest=max(runs,key=len,default=[])
            lag=lag_estimate(*zip(*longest)) if longest else None
            mean_compute=float(np.mean([r['compute_ms'] for r in rows]))
            summaries.append(dict(scenario=scenario,segment=segment,arm=name,frames=len(rows),
                output_coverage=len(valid)/len(rows),predicted_frames=predicted,resyncs=resync,
                stationary_std_width_units=float(np.mean(np.std(valid,axis=0))) if scenario=='stationary' and len(valid)>1 else None,
                jump_threshold_palm_units=jump_threshold,
                jump_count=sum(s>jump_threshold for s in steps),valid_step_pairs=len(steps),
                max_step_palm_units=max(steps,default=None),max_speed_palms_per_second=max(speeds,default=None),
                reappearance_step_palm_units=reappear,recovery_seconds=recovery,incomplete_recoveries=unfinished,
                raw_relative_wrist_x_lag=lag,
                rmse_to_supplied_truth_width_units=float(np.sqrt(np.mean(errors))) if errors else None,
                mean_arm_ms=float(np.mean(processing)),p95_arm_ms=float(np.percentile(processing,95)),
                mean_inference_ms=float(np.mean([r['inference_ms'] for r in rows])),
                mean_all_arms_compute_ms=mean_compute,
                all_arms_compute_fps=1000/mean_compute if mean_compute>0 else None))
    return {'warning':'Stability is not accuracy. Jump counts include genuine fast motion. Lag is relative to raw, not sensor-to-display latency.',
            'summaries':json_safe(summaries)}


def analyze(path,output_dir=None):
    report=summarize(load_frames(path))
    root=Path(output_dir) if output_dir else (Path(path) if Path(path).is_dir() else Path(path).parent)
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(report['summaries'][0]))
        writer.writeheader(); writer.writerows(report['summaries'])
    return report
