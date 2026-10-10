python3 - <<'PY'
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","pdf.fonttype":42})
INK="#1a1a1a"; AC="#26435c"; MU="#6a7480"; EDGE="#8494a3"
BANDS=["#dbe5ee","#e0eade","#f3e9da","#ebdde7","#e5e1ee"]
fig,ax=plt.subplots(figsize=(5.7,3.6)); ax.set_xlim(0,100); ax.set_ylim(0,58); ax.axis("off")
GX=11.0
def band(y,h,label,ci):
    ax.add_patch(Rectangle((GX,y),100-GX-1,h,fc=BANDS[ci],ec="none",alpha=.5,zorder=0))
    ax.add_patch(Rectangle((1,y),GX-2.2,h,fc=BANDS[ci],ec="none",alpha=.95,zorder=0))
    ax.text(1+(GX-2.2)/2,y+h/2,label,fontsize=6.0,color=AC,fontweight="bold",
            ha="center",va="center",rotation=90,zorder=1,linespacing=1.2)
def box(x,y,w,h,txt,fs=6.5,bold=False,hi=False,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.28,rounding_size=0.9",
        fc="white",ec=AC if hi else EDGE,lw=1.4 if hi else 0.8,ls=ls,zorder=2))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=INK,
            fontweight="bold" if bold else "normal",zorder=3,linespacing=1.35)
def arr(x1,y1,x2,y2,ls="-",lw=0.85):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=6.5,
        lw=lw,color=EDGE,ls=ls,zorder=4,shrinkA=0.4,shrinkB=0.4))
def line(x1,y1,x2,y2,lw=0.85):
    ax.plot([x1,x2],[y1,y2],color=EDGE,lw=lw,zorder=4,solid_capstyle="round")
def note(x,y,t,ha="left",fs=5.8,bg=None):
    ax.text(x,y,t,fontsize=fs,color=MU,style="italic",ha=ha,va="center",zorder=5,
            bbox=None if bg is None else dict(fc=bg,ec="none",pad=0.4))

band(46.5,11.0,"DATA",0); band(35.0,11.0,"REPRESENT.",1); band(27.0,7.5,"MODEL",2)
band(14.0,12.5,"CALIBRATION",3); band(1.5,12.0,"AUDIT",4)

y=50.6; h=4.4
box(GX+1.2,y,16.5,h,"Alibaba SMART\n475{,}056 drives".replace("{,}",","),fs=6.0,bold=True)
box(GX+20.5,y,14.5,h,"Daily records\n273 M rows",fs=6.0)
box(GX+38.0,y,14.5,h,"Labels\n$0<d_{\\mathrm{fail}}\\leq30$",fs=6.0)
box(GX+55.5,y,15.0,h,"Drive-level\nsplit 64/16/20",fs=6.0)
box(GX+74.0,y,13.0,h,"Hold-out\naxis",fs=6.0)
for a in (GX+17.7,GX+35.0,GX+52.5,GX+70.5): arr(a,y+h/2,a+2.8,y+h/2)
note(GX+1.2,47.6,"drop post-failure, unobservable tail, short history     drive key (model, serial)")
note(98.5,47.6,"vendor | wear stage",ha="right")

y=39.1
box(GX+1.2,y,17.5,h,"Windowing\n30 d, stride 30",fs=6.0,bold=True)
box(GX+22.0,y,19.5,h,"Normaliser\nz | rank | per-drive",fs=6.0)
box(GX+45.0,y,17.0,h,"Features\nlevel | dynamics",fs=6.0)
box(GX+65.5,y,21.5,h,"Common-16\n8 SMART IDs",fs=6.0)
for a in (GX+18.7,GX+41.5,GX+62.0): arr(a,y+h/2,a+3.3,y+h/2)
line(GX+62,50.6,GX+62,45.0); arr(GX+62,45.0,GX+62,43.5)
note(GX+1.2,36.2,"non-overlapping  |  label from LAST row  |  fit on TRAIN only",bg="#eef3ec")
note(98.5,36.2,"16 of 102 columns shared",ha="right")

y=29.0; hh=4.0
box(GX+15,y,25,hh,"Random Forest\nclass-weighted",fs=6.2,bold=True)
box(GX+46,y,24,hh,"scores $\\hat p(\\mathrm{fail}\\mid x)$",fs=6.5)
arr(GX+40,y+hh/2,GX+46,y+hh/2); line(GX+27,39.1,GX+27,33.5); arr(GX+27,33.5,GX+27,33.0)
BUS=25.8; line(GX+58,29.0,GX+58,BUS); line(GX+6,BUS,GX+80,BUS,lw=1.0)

y=17.6; w=15.6; gap=1.3; cols=[]
for i,(t,hi,ls) in enumerate([("Split\nmarginal",False,"-"),("Mondrian\nby class",True,"-"),
        ("Mondrian\nby group",False,"-"),("Mondrian\nclass $\\times$ group",True,"-"),
        ("Weighted\ndensity ratio",False,"--")]):
    x=GX+0.5+i*(w+gap); cols.append(x+w/2)
    arr(x+w/2,BUS,x+w/2,y+4.9); box(x,y,w,4.9,t,fs=6.0,bold=hi,hi=hi,ls=ls)
note(GX+0.5,15.2,"cell absent $\\rightarrow$ pooled fallback, flagged",fs=5.6,bg="#f5edf2")
note(98.5,15.2,r"$\hat q$ = $\lceil (n{+}1)(1-\alpha)\rceil$-th smallest score",ha="right",fs=5.6,bg="#f5edf2")

BUS2=12.2
for cx in cols: line(cx,17.6,cx,BUS2)
line(cols[0],BUS2,cols[-1],BUS2,lw=1.0)
y=5.6; hb=4.6
ev=[(GX+0.5,16.0,"Coverage\nmarginal",False),(GX+18.0,16.0,"Coverage\nper class",True),
    (GX+35.5,17.5,"Set size &\ncomposition",True),(GX+54.5,15.5,"Cell sizes,\nbootstrap CIs",False),
    (GX+71.5,15.5,"Stratification\nvalidator",True)]
for x,ww,t,hi in ev:
    arr(x+ww/2,BUS2,x+ww/2,y+hb); box(x,y,ww,hb,t,fs=6.0,bold=hi,hi=hi)
note(50,2.3,"the validator rejects a covariate as a calibration dimension when its effect direction is inconsistent across populations",ha="center",fs=5.8)
ax.text(98.5,56.8,"bold outline: contributes a reported finding",fontsize=5.5,color=AC,ha="right",style="italic")
fig.tight_layout(pad=0.12); fig.savefig("fig_architecture.pdf")
print("ok")
PY
