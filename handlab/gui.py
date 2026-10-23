"""Tk main thread for widgets; detector runs in a separate single capture thread."""
import logging
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import Config


SCENARIOS=('unlabelled','stationary','slow','fast','fingers','partial','occlusion','reappearance')


class Worker(threading.Thread):
    def __init__(self,source,model,config,output,events,commands):
        super().__init__(daemon=True)
        self.source,self.model,self.config,self.output=source,model,config,output
        self.events,self.commands=events,commands
        self.stop_requested=threading.Event()

    def send(self,kind,payload):
        try:self.events.put_nowait((kind,payload))
        except queue.Full:
            # Render backpressure never discards data from the recorder.
            if kind in ('error','ended','recording'):
                try:self.events.get_nowait()
                except queue.Empty:pass
                self.events.put_nowait((kind,payload))

    def run(self):
        from .capture import Capture
        from .tracker import HandTracker
        from .experiment import Experiment
        from .recording import Recorder, session_path
        from .render import comparison_image
        import cv2
        rec=None; e=Experiment(self.config); scenario='unlabelled'; arm='custom'; overlay=False
        previous_wall=None; processed_fps=0.
        try:
            with Capture(self.source) as source,HandTracker(self.model) as tracker:
                self.send('ready',str(self.source))
                while not self.stop_requested.is_set():
                    while True:
                        try: cmd,value=self.commands.get_nowait()
                        except queue.Empty: break
                        if cmd=='stop': self.stop_requested.set(); break
                        if cmd=='config': e.configure(value,reset_segment=True)
                        if cmd=='scenario': scenario=value
                        if cmd=='arm': arm=value
                        if cmd=='overlay': overlay=value
                        if cmd=='record':
                            if value and rec is None:
                                folder=session_path(self.output)
                                rec=Recorder(folder,{'source':str(self.source),'model_path':str(self.model)},e.pipelines)
                                self.send('recording',str(folder))
                            elif not value and rec:
                                rec.close(); self.send('recording',f'Saved: {rec.directory}')
                                rec=None
                    if self.stop_requested.is_set(): break
                    packet=source.read()
                    if packet is None: break
                    t,bgr=packet
                    tick=time.perf_counter()
                    obs=tracker.detect(bgr,t)
                    inference=(time.perf_counter()-tick)*1000
                    f=e.step(obs,t,aspect=bgr.shape[0]/bgr.shape[1],scenario=scenario,inference_ms=inference)
                    if rec: rec.write(f)
                    now=time.perf_counter()
                    if previous_wall is not None:
                        current=1/max(now-previous_wall,1e-6)
                        processed_fps=current if not processed_fps else .8*processed_fps+.2*current
                    previous_wall=now
                    if self.events.qsize()<2:
                        image=comparison_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),f,arm,overlay)
                        self.send('frame',(image,f['outputs'][arm],processed_fps,f['compute_ms'],rec is not None))
        except Exception as exc:
            logging.exception('Worker failed')
            self.send('error',str(exc))
        finally:
            if rec:
                rec.close(); self.send('recording',f'Saved: {rec.directory}')
            self.send('ended',None)


class App:
    def __init__(self,root,model,config,output):
        self.root=root;self.model=model;self.config=config;self.output=output
        self.worker=None;self.events=queue.Queue(maxsize=4);self.commands=queue.Queue()
        self.photo=None;self.pending=None
        root.title('HandLab | RGB hand tracking experiments')
        root.minsize(850,680)
        panel=ttk.Frame(root,padding=10);panel.pack(fill='both',expand=True)
        actions=ttk.Frame(panel);actions.pack(fill='x')
        ttk.Button(actions,text='Start webcam',command=lambda:self.start(0)).pack(side='left')
        ttk.Button(actions,text='Open video',command=self.open_video).pack(side='left',padx=5)
        ttk.Button(actions,text='Stop',command=self.stop).pack(side='left')
        self.record=tk.BooleanVar(value=False)
        ttk.Checkbutton(actions,text='Record data',variable=self.record,command=lambda:self.command('record',self.record.get())).pack(side='left',padx=10)
        ttk.Label(actions,text='Scenario').pack(side='left')
        self.scenario=tk.StringVar(value=SCENARIOS[0]); box=ttk.Combobox(actions,textvariable=self.scenario,values=SCENARIOS,state='readonly',width=14)
        box.pack(side='left',padx=5);box.bind('<<ComboboxSelected>>',lambda _:self.command('scenario',self.scenario.get()))
        choices=ttk.Frame(panel);choices.pack(fill='x',pady=5)
        self.arm=tk.StringVar(value='custom')
        ttk.Label(choices,text='View').pack(side='left')
        select=ttk.Combobox(choices,textvariable=self.arm,values=('raw','oef','corrected','full','custom'),state='readonly',width=12)
        select.pack(side='left');select.bind('<<ComboboxSelected>>',lambda _:self.command('arm',self.arm.get()))
        self.overlay=tk.BooleanVar(value=False)
        ttk.Checkbutton(choices,text='Raw overlay on right',variable=self.overlay,command=lambda:self.command('overlay',self.overlay.get())).pack(side='left',padx=10)
        self.checks={}
        checks=ttk.LabelFrame(panel,text='Custom comparison switches',padding=5);checks.pack(fill='x',pady=3)
        for key,label in (('oef_enabled','One Euro'),('detection_enabled','Anomaly detection'),
                          ('correction_enabled','Selective correction'),('ema_enabled','EMA recovery')):
            var=tk.BooleanVar(value=getattr(config,key));self.checks[key]=var
            ttk.Checkbutton(checks,text=label,variable=var,command=self.schedule_config).pack(side='left',padx=8)
        sliders=ttk.LabelFrame(panel,text='Filter settings (applied after slider release)',padding=5)
        sliders.pack(fill='x',pady=4)
        self.scales={}
        for key,label,minimum,maximum,res in [('min_cutoff','OEF min cutoff',.1,10,.1),('beta','OEF beta',0,10,.1),
            ('d_cutoff','OEF derivative cutoff',.1,10,.1),('ema_tau','EMA time constant (s)',.015,.6,.005),
            ('recovery_timeout','Recovery timeout (s)',.1,2,.05)]:
            row=ttk.Frame(sliders);row.pack(fill='x')
            ttk.Label(row,text=label,width=27).pack(side='left')
            variable=tk.DoubleVar(value=getattr(config,key));self.scales[key]=variable
            scale=tk.Scale(row,from_=minimum,to=maximum,resolution=res,orient='horizontal',showvalue=True,
                           variable=variable,command=lambda _value:self.schedule_config())
            scale.pack(side='left',fill='x',expand=True)
        self.status=tk.StringVar(value='Idle | Start webcam or choose a video')
        ttk.Label(panel,textvariable=self.status).pack(fill='x',pady=5)
        self.preview=ttk.Label(panel,anchor='center',text='MediaPipe raw | Corrected')
        self.preview.pack(fill='both',expand=True)
        self.path=tk.StringVar(value='Recording is off')
        ttk.Label(panel,textvariable=self.path).pack(fill='x')
        ttk.Button(panel,text='Save current settings JSON',command=self.save_config).pack(anchor='e')
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.poll()

    def current_config(self):
        return Config.from_dict({**self.config.to_dict(),**{k:v.get() for k,v in self.checks.items()},
                                  **{k:v.get() for k,v in self.scales.items()}})

    def schedule_config(self):
        if self.pending is not None: self.root.after_cancel(self.pending)
        self.pending=self.root.after(250,self.apply_config)

    def apply_config(self):
        self.pending=None
        try:self.config=self.current_config()
        except ValueError as exc:
            self.status.set(str(exc));return
        self.command('config',self.config)

    def save_config(self):
        try:
            path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
            if path:self.current_config().save(path)
        except (OSError,ValueError) as exc:messagebox.showerror('Save failed',str(exc))

    def command(self,kind,value):
        if self.worker and self.worker.is_alive(): self.commands.put((kind,value))

    def open_video(self):
        path=filedialog.askopenfilename(filetypes=[('Videos','*.mp4 *.avi *.mov *.mkv'),('All files','*.*')])
        if path:self.start(path)

    def start(self,source):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo('HandLab','Stop the current source first.');return
        self.events=queue.Queue(maxsize=4);self.commands=queue.Queue()
        self.worker=Worker(source,self.model,self.current_config(),self.output,self.events,self.commands)
        self.worker.start();self.status.set('Opening source...')

    def stop(self):
        if self.worker:self.worker.stop_requested.set()

    def poll(self):
        while True:
            try:kind,value=self.events.get_nowait()
            except queue.Empty:break
            if kind=='frame':
                from PIL import ImageTk
                image,out,fps,compute,recording=value
                limit=max(250,self.root.winfo_width()-22)
                if image.width>limit:image.thumbnail((limit,800))
                self.photo=ImageTk.PhotoImage(image)
                self.preview.configure(image=self.photo,text='')
                self.status.set(f"{out['state']} | {out['reason']} | {fps:.1f} FPS | {compute:.1f} ms | recording={recording}")
            elif kind=='recording':self.path.set(value)
            elif kind=='error':self.status.set(f'Error: {value}');messagebox.showerror('HandLab',value)
            elif kind=='ended':self.record.set(False);self.status.set('Stopped / video ended')
            elif kind=='ready':self.status.set(f'Running {value}')
        self.root.after(30,self.poll)

    def close(self):
        self.stop()
        # Wait for recorder finally block to flush CSV/JSONL before exiting.
        if self.worker and self.worker.is_alive():
            self.root.after(50,self.close)
            return
        self.root.destroy()


def launch(model,config,output):
    try:
        root=tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError('GUI 화면을 열 수 없습니다. 데스크톱 DISPLAY와 Tk 설치를 확인하세요.') from exc
    App(root,model,config,output)
    root.mainloop()
