"""Entry point: GUI, deterministic synthetic replay, offline video, analysis."""
import argparse
import logging
from pathlib import Path
import sys
import time

from .config import Config


def process_source(source,model,config,scenario,record_root):
    from .capture import Capture
    from .experiment import Experiment
    from .tracker import HandTracker
    from .recording import Recorder, session_path
    from .evaluation import analyze
    from .render import comparison_image
    import cv2
    e=Experiment(config)
    folder=session_path(record_root)
    with Capture(source) as capture, HandTracker(model) as tracker, Recorder(folder,
        {'source':str(source),'model_path':str(model),'input_type':'webcam' if isinstance(source,int) else 'video'},
        e.pipelines) as rec:
        while True:
            packet=capture.read()
            if packet is None: break
            t,bgr=packet
            start=time.perf_counter()
            obs=tracker.detect(bgr,t)
            inference=(time.perf_counter()-start)*1000
            frame=e.step(obs,t,aspect=bgr.shape[0]/bgr.shape[1],scenario=scenario,inference_ms=inference)
            rec.write(frame)
            if e.frame%30==0:
                logging.info('%d frames; %s; %.1f ms/frame; %.1f FPS',e.frame,
                             frame['outputs']['custom']['state'],frame['compute_ms'],
                             1000/frame['compute_ms'] if frame['compute_ms'] else 0)
    report=analyze(folder)
    print(f'{folder} | {e.frame} frames | {len(report["summaries"])} scenario/arm summaries')
    return folder


def run_synthetic(args):
    from .synthetic import sequence
    from .tracker import Observation
    from .experiment import Experiment
    from .recording import Recorder,session_path
    from .evaluation import analyze
    e=Experiment(Config.load(args.config)); folder=session_path(args.output)
    with Recorder(folder,{'source':'synthetic','seed':args.seed,'scenario':args.scenario},e.pipelines) as rec:
        for t,x,truth in sequence(args.scenario,args.frames,args.fps,args.seed):
            rec.write(e.step(Observation(x),t,scenario=args.scenario,truth=truth))
    analyze(folder)
    print(f'{folder} | synthetic frames: {args.frames}')


def main(argv=None):
    parser=argparse.ArgumentParser(description='MediaPipe HandLab (CPU, 21 landmarks)')
    parser.add_argument('--verbose',action='store_true')
    sub=parser.add_subparsers(dest='action',required=True)
    gui=sub.add_parser('gui',help='Tk GUI, webcam or video')
    gui.add_argument('--model',default='models/hand_landmarker.task')
    gui.add_argument('--config'); gui.add_argument('--output',default='experiments')
    video=sub.add_parser('video',help='headless video replay with five comparison arms')
    video.add_argument('path'); video.add_argument('--model',default='models/hand_landmarker.task')
    video.add_argument('--config'); video.add_argument('--scenario',default='unlabelled')
    video.add_argument('--output',default='experiments')
    cam=sub.add_parser('camera',help='headless webcam capture; Ctrl+C to stop')
    cam.add_argument('--device',type=int,default=0);cam.add_argument('--model',default='models/hand_landmarker.task')
    cam.add_argument('--config');cam.add_argument('--scenario',default='unlabelled')
    cam.add_argument('--output',default='experiments')
    synth=sub.add_parser('synthetic',help='replay deterministic artificial landmarks')
    synth.add_argument('--scenario',choices=('stationary','slow','fast','fingers','joint_jump','whole_jump','partial','occlusion'),default='stationary')
    synth.add_argument('--frames',type=int,default=240);synth.add_argument('--fps',type=float,default=30.)
    synth.add_argument('--seed',type=int,default=7);synth.add_argument('--config')
    synth.add_argument('--output',default='experiments')
    analysis=sub.add_parser('analyze');analysis.add_argument('session')
    model=sub.add_parser('download-model');model.add_argument('--output',default='models/hand_landmarker.task')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        if args.action=='download-model':
            from .model import download_model
            print(download_model(args.output))
        elif args.action=='synthetic':
            if args.frames<=0 or args.fps<=0: raise ValueError('frames와 fps는 양수여야 합니다.')
            run_synthetic(args)
        elif args.action=='analyze':
            from .evaluation import analyze
            from .recording import load_frames
            import json
            print(json.dumps(analyze(args.session),ensure_ascii=False,indent=2))
        elif args.action=='gui':
            from .gui import launch
            launch(args.model,Config.load(args.config),args.output)
        elif args.action=='video':
            process_source(args.path,args.model,Config.load(args.config),args.scenario,args.output)
        elif args.action=='camera':
            process_source(args.device,args.model,Config.load(args.config),args.scenario,args.output)
    except (OSError,RuntimeError,ValueError,ImportError,KeyboardInterrupt) as exc:
        print(f'HandLab: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
