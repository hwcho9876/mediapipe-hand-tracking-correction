"""Append-only JSONL ground data and wide CSV for spreadsheet analysis."""
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np


def json_safe(value):
    if isinstance(value,np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value,np.generic):
        return json_safe(value.item())
    if isinstance(value,dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def environment_info():
    versions={}
    for name in ('numpy','mediapipe','opencv-python','Pillow'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return dict(python=platform.python_version(),platform=platform.platform(),packages=versions)


class Recorder:
    def __init__(self, directory, metadata, names):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.names=list(names)
        self.json_file=(self.directory/'frames.jsonl').open('w',encoding='utf-8')
        self.csv_file=(self.directory/'frames.csv').open('w',newline='',encoding='utf-8')
        columns=['frame','t','scenario','segment','inference_ms','postprocess_ms','compute_ms']
        for name in self.names:
            columns += [f'{name}_{k}' for k in ('state','reason','predicted','resynced','processing_ms')]
            columns += [f'{name}_{i}_{axis}' for i in range(21) for axis in 'xyz']
        self.csv=csv.DictWriter(self.csv_file,fieldnames=columns)
        self.csv.writeheader()
        meta=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
                  coordinate_system='image-width units: x, y*height/width, z; z is wrist-relative',
                  environment=environment_info(),**metadata)
        (self.directory/'metadata.json').write_text(json.dumps(json_safe(meta),ensure_ascii=False,indent=2),encoding='utf-8')
        self.count=0

    def write(self, frame):
        safe=json_safe(frame)
        self.json_file.write(json.dumps(safe,ensure_ascii=False,allow_nan=False)+'\n')
        row={k:safe[k] for k in ('frame','t','scenario','segment','inference_ms','postprocess_ms','compute_ms')}
        for name,output in safe['outputs'].items():
            for key in ('state','reason','predicted','resynced','processing_ms'):
                row[f'{name}_{key}']=output[key]
            if output['points'] is not None:
                for i,point in enumerate(output['points']):
                    for axis,value in zip('xyz',point):
                        row[f'{name}_{i}_{axis}']=value
        self.csv.writerow(row)
        self.count+=1
        if self.count%30==0:
            self.json_file.flush(); self.csv_file.flush()

    def close(self):
        self.json_file.close(); self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


def session_path(root='experiments'):
    return Path(root)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def load_frames(path):
    path=Path(path)
    path=path/'frames.jsonl' if path.is_dir() else path
    with path.open(encoding='utf-8') as f:
        for line_number,line in enumerate(f,1):
            try:
                value=json.loads(line)
                if 't' not in value or 'outputs' not in value:
                    raise ValueError('프레임 필드 누락')
                yield value
            except (json.JSONDecodeError,ValueError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
