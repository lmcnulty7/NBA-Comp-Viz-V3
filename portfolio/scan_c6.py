import subprocess, json, sys, numpy as np
src = sys.argv[1]; w, h = int(sys.argv[2]), int(sys.argv[3])
raw = subprocess.run(["/opt/homebrew/bin/ffmpeg","-v","error","-i",src,"-f","rawvideo","-pix_fmt","bgr24","-"],capture_output=True).stdout
n = len(raw)//(w*h*3); fr = np.frombuffer(raw,np.uint8).reshape(n,h,w,3)
def band_frac(region):
    b,g,r = region[...,0].astype(int),region[...,1].astype(int),region[...,2].astype(int)
    m = (b<45)&(g<45)&(r>80)&(r<165)
    return m.mean(axis=(1,2))
bot = band_frac(fr[:, h-26:h, :, :]); top = band_frac(fr[:, 0:24, :, :])
bot_idx = [i for i in range(n) if bot[i] > 0.5]; top_idx = [i for i in range(n) if top[i] > 0.5]
print("frames", n, "bottom-band frames", len(bot_idx), "top-band frames", len(top_idx))
print("bottom idx", bot_idx)
if len(sys.argv) > 4:
    inv = json.load(open(sys.argv[4])); core = inv["core"]; exp = [core.index(f) for f in inv["invalid"]]
    print("match wrapper invalid indices:", bot_idx == exp)
