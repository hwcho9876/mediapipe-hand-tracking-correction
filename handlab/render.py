"""Render without importing cv2: works for synthetic preview and a lightweight Tk UI."""
import numpy as np
from PIL import Image, ImageDraw
from .geometry import EDGES

COLORS={'raw':(0,190,255),'oef':(245,190,70),'corrected':(190,130,255),'full':(60,230,150),'custom':(60,230,150)}


def draw_hand(image,points,color):
    if points is None:
        return
    draw=ImageDraw.Draw(image)
    w,h=image.size
    p=np.asarray(points)
    if p.shape!=(21,3): return
    coords=[(int(np.clip(x[0]*w,-2*w,3*w)),int(np.clip(x[1]*w,-2*h,3*h))) if np.isfinite(x).all() else None for x in p]
    for a,b in EDGES:
        if coords[a] and coords[b]: draw.line([coords[a],coords[b]],fill=color,width=3)
    for pt in coords:
        if pt:
            x,y=pt; draw.ellipse((x-3,y-3,x+3,y+3),fill=color)


def comparison_image(rgb,frame,arm='custom',overlay=False):
    base=Image.fromarray(rgb) if isinstance(rgb,np.ndarray) else rgb.copy()
    left=base.copy(); right=base.copy()
    draw_hand(left,frame['raw'],COLORS['raw'])
    if overlay: draw_hand(right,frame['raw'],COLORS['raw'])
    out=frame['outputs'][arm]
    draw_hand(right,out['points'],COLORS.get(arm,COLORS['full']))
    result=Image.new('RGB',(base.width*2,base.height+40),(18,24,32))
    result.paste(left,(0,40)); result.paste(right,(base.width,40))
    d=ImageDraw.Draw(result)
    d.text((12,12),'MediaPipe raw',fill=COLORS['raw'])
    label=f"{arm} | {out['state']} | {out['reason']}"
    d.text((base.width+12,12),label,fill=COLORS.get(arm,COLORS['full']))
    return result
