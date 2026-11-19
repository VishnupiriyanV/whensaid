"""Renders assets/demo.gif from real `whensaid` output. Needs pillow.
WHENSAID_HOME must point at an index (the README one is 65 h of public-domain audio)."""
import shutil, subprocess, sys, textwrap, os
from PIL import Image, ImageDraw, ImageFont
W_BIN = shutil.which("whensaid")
env = dict(os.environ)
OUT = sys.argv[1] if len(sys.argv) > 1 else "assets/demo.gif"
cmds=[["status"],
      ["search","two men arguing about what day of the month it is","-n","1"],
      ["search","a mouse tribe claiming rights over everyone else","-n","1"],
      ["search","how to feed and maintain a sourdough starter","-n","1"]]
COLS=104; ROWS=20; FS=16
font=ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf",FS)
bold=ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf",FS)
cw=int(font.getlength("M")); lh=FS+6; PAD=14
IMG=(COLS*cw+2*PAD, ROWS*lh+2*PAD+22)
BG=(24,26,31); FG=(220,223,228); GREEN=(120,200,120); AMBER=(230,180,80); DIM=(130,136,146); BLUE=(110,170,240)
def shown(cmd):
    return "$ whensaid "+" ".join(f'"{a}"' if " " in a else a for a in cmd)
def wrap(line, first_color):
    return [(l,first_color) for l in textwrap.wrap(line,COLS-2,subsequent_indent="  ") or [""]]
def run(cmd):
    out=subprocess.run([W_BIN]+cmd,capture_output=True,text=True,env=env).stdout.rstrip("\n").split("\n")
    rows=[]
    for l in out:
        col=GREEN if l.startswith("strong") else AMBER if l.startswith(("weak","only weak","nothing")) else FG
        rows+=wrap(l,col)
    return rows
frames=[];durs=[]
def draw(lines, cursor=True):
    im=Image.new("RGB",IMG,BG); d=ImageDraw.Draw(im)
    for i,c in enumerate((255,95,86),0): d.ellipse((PAD+i*20,8,PAD+i*20+11,19),fill=[(255,95,86),(255,189,46),(39,201,63)][i])
    y=PAD+22
    for text,col in lines[-ROWS:]:
        d.text((PAD,y),text,font=bold if text.startswith("$") else font,fill=BLUE if text.startswith("$") else col); y+=lh
    if cursor and lines:
        t=lines[-1][0]; d.rectangle((PAD+len(t)*cw,y-lh+2,PAD+len(t)*cw+cw-2,y-4),fill=FG)
    return im
def add(im,ms): frames.append(im); durs.append(ms)
screen=[]
for cmd in cmds:
    typed=shown(cmd)
    if len(screen)>ROWS-10: screen=[]
    base=list(screen)
    for n in range(0,len(typed)+1,3):
        add(draw(base+[(typed[:n],BLUE)]),45)
    add(draw(base+[(typed,BLUE)]),500)
    out=run(cmd)
    screen=base+[(typed,BLUE)]
    add(draw(screen+[("",FG)]),350)
    for i in range(0,len(out),2):
        screen_t=base+[(typed,BLUE)]+out[:i+2]
        add(draw(screen_t,cursor=False),90)
    screen=base+[(typed,BLUE)]+out+[("",FG)]
    add(draw(screen),2800)
frames[0].save(OUT,save_all=True,append_images=frames[1:],duration=durs,loop=0,optimize=True)
print(len(frames),sum(durs)/1000,"s")
