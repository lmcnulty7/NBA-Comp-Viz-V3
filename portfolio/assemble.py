#!/opt/anaconda3/bin/python
"""Fill template.html placeholders with data: URIs, numbers and generated charts."""
import base64, json, os, re, subprocess, sys, datetime, io
from pathlib import Path
from PIL import Image

SC = Path(__file__).resolve().parent          # portfolio/
OUT_DIR = SC / 'build'                          # gitignored; regenerable
OUT_DIR.mkdir(exist_ok=True)
TEMPLATE = SC / 'template.html'
NUMBERS = SC / 'numbers.json'
FFMPEG = '/opt/homebrew/bin/ffmpeg'
FFPROBE = '/opt/homebrew/bin/ffprobe'
CLIPS = {
 'c1_court_lock': SC/'clips/c1_court_lock.mp4',
 'c2_gate_deadball': SC/'clips/c2_gate_deadball.mp4',
 'c3_possession_sidebyside': SC/'clips/c3_possession_sidebyside.mp4',
 'c4_pixels_to_feet_curry3': SC/'clips/c4_pixels_to_feet_curry3.mp4',
 'c5_matchup_topdown': SC/'clips/c5_matchup_topdown.mp4',
 'c6_excluded_banner': SC/'clips/c6_excluded_banner.mp4',
 'c7_held_1998': SC/'clips/c7_held_1998.mp4',
}
STILLS = {s['id']: s for s in json.load(open(SC/'img/stills_manifest.json'))}
N = json.load(open(NUMBERS))
# derived (from JSON fields, not typed)
m = N['matchup_curry_q1_span_0']; m['frames_sampled'] = m['frames_used'] + m['frames_excluded_team_gt5']
_years = sorted(k[:4] for k in N['crossval_per_game']['per_game'])
N['corpus_years'] = {'min': _years[0], 'max': _years[-1]}
assert N['possessions']['basket_n'] == N['possessions']['offense_n']
N['possessions']['n_unclear'] = N['possessions']['n_labeled'] - N['possessions']['basket_n']
REPO = Path(N['_meta']['repo_root'])
import pypdf
N['paper_pages'] = len(pypdf.PdfReader(REPO/'paper/main.pdf').pages)
_spans = json.load(open(REPO/'data/tracking/gsw_cle_2017f_g5_s03_possessions.json'))['spans']
_sp = [p for p in _spans if p.get('set_start_frame') == N['record_card_f33816']['set_start_frame']]
assert len(_sp) == 1 and _sp[0]['core_end_frame'] == N['record_card_f33816']['core_end_frame']
N['record_card_f33816']['end_frame'] = _sp[0]['end_frame']

def probe(path):
    out = subprocess.run([FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries',
        'stream=nb_read_frames,width,height,duration','-of','json',str(path)],capture_output=True,text=True,check=True).stdout
    st = json.loads(out)['streams'][0]
    return int(st['nb_read_frames']), int(st['width']), int(st['height']), float(st['duration'])

def b64(path, mime):
    data = Path(path).read_bytes()
    return 'data:%s;base64,%s' % (mime, base64.b64encode(data).decode('ascii')), len(data)

def poster(cid, path, dur, width, q):
    """Extract the mid-frame with ffmpeg, encode WebP at the given width/quality; returns (data_uri, w, h)."""
    png = SC/('posters/%s_%d.png' % (cid, width)); webp = SC/('posters/%s_%d.webp' % (cid, width))
    png.parent.mkdir(exist_ok=True)
    for f in (png, webp):
        if f.exists(): f.unlink()
    subprocess.run([FFMPEG,'-v','error','-y','-ss',str(dur/2),'-i',str(path),'-frames:v','1',str(png)],check=True)
    im = Image.open(png).convert('RGB')
    if im.width > width: im = im.resize((width, round(im.height*width/im.width)), Image.LANCZOS)
    im.save(webp,'WEBP',quality=q,method=6)
    assert webp.stat().st_size > 1000, 'poster too small: %s' % webp
    uri, nbytes = b64(webp,'image/webp')
    return uri, im.width, im.height, nbytes

def fmt(v, spec):
    def sign(s): return re.sub(r'(?<![\w/])-(?=\d)', '−', s)
    if spec is None or spec == 'raw':
        if isinstance(v, float):
            s = ('%f' % v).rstrip('0').rstrip('.')
        else: s = str(v)
        return sign(s)
    if spec == 'pct': return sign('%.1f' % (v*100))
    if spec == 'pct0': return sign('%d' % round(v*100))
    if spec == 'int': return sign('%d' % round(v))
    if spec == 'comma': return sign('{:,}'.format(int(round(v))))
    if spec == 'f1': return sign('%.1f' % v)
    if spec == 'f2': return sign('%.2f' % v)
    if spec == 'f3': return sign('%.3f' % v)
    if spec == 'signed1': return sign(('%+.1f' % v)).replace('+', '+')
    raise KeyError('unknown fmt '+spec)

def lookup(path):
    cur = N; segs = path.split('.'); i = 0
    while i < len(segs):
        seg = segs[i]
        if isinstance(cur, list): cur = cur[int(seg)]
        elif seg in cur: cur = cur[seg]
        elif i+1 < len(segs) and seg+'.'+segs[i+1] in cur: cur = cur[seg+'.'+segs[i+1]]; i += 1
        else: raise KeyError(path)
        i += 1
    return cur

def esc(s): return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def nfill(mo):
    path, _, spec = mo.group(1).partition('|')
    return esc(fmt(lookup(path), spec or None))

# ---- charts -----------------------------------------------------------------
def minus(s): return s.replace('-', '−')

def chart_f1():
    g = N['credit']['gsw']; lo, hi = g['ci95']; pt = g['credit_per_100']
    x0, x1, p0, p1 = -30.0, 15.0, 6.0, 94.0
    X = lambda v: '%.2f%%' % (p0 + (v-x0)/(x1-x0)*(p1-p0))
    y = 40; ax = 62
    ticks = ''.join('<line class="ax" x1="%s" x2="%s" y1="%d" y2="%d"/><text class="mu" x="%s" y="%d" text-anchor="middle">%s</text>' % (X(t),X(t),ax,ax+5,X(t),ax+18,minus(str(t))) for t in (-30,-20,-10,0,10))
    return ('<div class="sub">GSW · n = %d · %d games</div>' % (g['n_possessions'], g['n_games'])
        + '<svg height="84" role="img" aria-label="Interval chart: GSW credit per 100 possessions, point %s, 95%% CI %s to %s, n = %d, %d games">' % (minus(str(pt)),minus(str(lo)),minus(str(hi)),g['n_possessions'],g['n_games'])
        + '<line class="ax" x1="%s" x2="%s" y1="%d" y2="%d"/>' % (X(x0), X(x1), ax, ax) + ticks
        + '<line class="zero" x1="%s" x2="%s" y1="14" y2="%d"/><text class="mu" x="%s" y="10" text-anchor="middle">zero</text>' % (X(0),X(0),ax,X(0))
        + '<line class="bar-h" x1="%s" x2="%s" y1="%d" y2="%d"/>' % (X(lo),X(hi),y,y)
        + '<circle class="dot-h" cx="%s" cy="%d" r="5"/>' % (X(pt),y)
        + '<text class="mu" x="%s" dx="-9" y="%d" text-anchor="end">%s</text>' % (X(lo), y+4, minus('%.1f' % lo))
        + '<text class="mu" x="%s" dx="9" y="%d">+%.1f</text>' % (X(hi), y+4, hi)
        + '<text x="%s" y="%d" text-anchor="middle">%s</text>' % (X(pt), y-12, minus('%.1f' % pt))
        + '</svg>')

def chart_f5():
    rows = N['label_audit']['f5_chart_rows']; agg = N['label_audit']['overall']['accuracy']
    X = lambda v: '%.2f%%' % (2.0 + v*96.0)
    out = ['<div class="grid" role="img" aria-label="Dot-and-interval chart of audit accuracy by stratum with Wilson 95%% intervals; aggregate %.1f%% is not a meaningful number because it mixes bands">' % (agg*100)]
    out.append('<span></span><div class="ticks">' + ''.join('<span style="left:%s">%d%%</span>' % (X(t), round(t*100)) for t in (0,.25,.5,.75,1)) + '</div><span></span>')
    for r in rows:
        lo, hi = r['wilson95']; m = r['band'] == 'agreement'
        cls_bar = 'bar-m' if m else 'bar-x'; cls_dot = 'dot-m' if m else 'dot-x'
        out.append('<span class="lab"><i class="g %s"></i>%s</span>' % ('g-m' if m else 'g-u', esc(r['stratum'])))
        out.append('<svg height="22" aria-hidden="true">'
            + ''.join('<line class="ax" x1="%s" x2="%s" y1="0" y2="22"/>' % (X(t), X(t)) for t in (0,.25,.5,.75,1))
            + '<line class="ref" x1="%s" x2="%s" y1="0" y2="22"/>' % (X(agg), X(agg))
            + '<line class="%s" x1="%s" x2="%s" y1="11" y2="11"/>' % (cls_bar, X(lo), X(hi))
            + '<circle class="%s" cx="%s" cy="11" r="4.5"/>' % (cls_dot, X(r['accuracy']))
            + '</svg>')
        out.append('<span class="val">%.3f · n=%d</span>' % (r['accuracy'], r['n_judged']))
    out.append('</div>')
    out.append('<div class="cap">dot = accuracy on the audited sample · bar = Wilson 95% interval · dashed line = the aggregate, explained below</div>')
    return ''.join(out)

def chart_numline():
    g = N['credit']['gsw']; lo, hi = g['ci95']; pt = g['credit_per_100']
    x0, x1, px0, px1 = -25.0, 10.0, 14, 346
    X = lambda v: px0 + (v-x0)/(x1-x0)*(px1-px0)
    y = 26
    ticks = ''.join('<line class="ax" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/><text x="%.1f" y="46" text-anchor="middle">%s</text>' % (X(t),X(t),y+4,y+8,X(t),minus(str(t))) for t in (-20,-10,0,10))
    return ('<svg viewBox="0 0 360 50" role="img" aria-label="Number line: GSW credit per 100 possessions, point %s, interval %s to +%s, zero marked">' % (minus(str(pt)),minus(str(lo)),hi)
        + '<line class="ax" x1="14" x2="346" y1="%d" y2="%d"/>' % (y+4,y+4) + ticks
        + '<line class="zero" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (X(0),X(0),y-10,y+10)
        + '<line class="bar" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (X(lo),X(hi),y,y)
        + '<circle class="dot" cx="%.1f" cy="%d" r="4"/>' % (X(pt),y)
        + '<text x="%.1f" y="11" text-anchor="middle">%s [%s, +%s] per 100 · n = %d</text>' % (X(pt), minus(str(pt)), minus(str(lo)), hi, g['n_possessions'])
        + '</svg>')

# ---- assemble ---------------------------------------------------------------
tpl = TEMPLATE.read_text()
report = {'clips': {}, 'stills': {}}
video_uri, poster_uri, still_uri, dims = {}, {}, {}, {}
for cid, path in CLIPS.items():
    frames, w, h, dur = probe(path)
    raw = path.stat().st_size
    uri, nbytes = b64(path, 'video/mp4')
    assert frames == round(dur*24), (cid, frames, dur)
    video_uri[cid] = uri
    p_uri, pw, ph, pb = poster(cid, path, dur, 480, 75)
    s_uri, sw, sh, sb = poster(cid, path, dur, w, 80)
    if cid in ('c1_court_lock', 'c3_possession_sidebyside'): p_uri, pb = s_uri, sb   # rendered at up to 1120/1024 px: native-width poster
    poster_uri[cid], still_uri[cid], dims[cid] = p_uri, s_uri, (w, h, sw, sh)
    report['clips'][cid] = {'raw_bytes': raw, 'base64_bytes': len(uri) - len('data:video/mp4;base64,'), 'frames': frames, 'w': w, 'h': h, 'dur': round(dur, 3), 'poster_bytes': pb, 'still_bytes': sb}
img_uri = {}
for sid, s in STILLS.items():
    uri, nbytes = b64(s['out'], 'image/webp'); img_uri[sid] = uri
    report['stills'][sid] = {'bytes': nbytes, 'base64_bytes': len(uri) - len('data:image/webp;base64,')}

def fill(html, stills_mode):
    html = html.replace('{{CHART:F1}}', chart_f1()).replace('{{CHART:F5}}', chart_f5()).replace('{{CHART:NUMLINE}}', chart_numline())
    html = html.replace('{{BUILD_DATE}}', datetime.date.today().isoformat())
    _led = re.search(r'<div class="ledger" id="ledger">.*?</table>', html, re.S).group(0)
    html = html.replace('{{LEDGER_ROWS}}', str(_led.count('<tr><td>')))
    html = re.sub(r'\{\{N:([^}]+)\}\}', nfill, html)
    if stills_mode:
        def repl(mo):
            cid = mo.group(1)
            if cid == 'c5_matchup_topdown': return mo.group(0)
            w, h, sw, sh = dims[cid]
            return ('<img class="still" id="v-%s" src="{{STILL:%s}}" width="%d" height="%d" alt="Still frame from the %s clip">'
                    '<span class="badge" style="top:auto;bottom:10px">still frame — clip available on request</span>') % (cid[:2], cid, sw, sh, cid.replace('_',' '))
        html = re.sub(r'<!--clip:([a-z0-9_]+)-->.*?<!--/clip-->', repl, html, flags=re.S)
        # a still has no playback: drop the progress line for every replaced clip
        html = re.sub(r'<div class="prog" aria-hidden="true"><i id="p-c[1-46-7]"></i></div>\s*', '', html)
        assert html.count('<span class="rc-live-note">Rows brighten as the clip plays.</span>') == 1
        html = html.replace('<span class="rc-live-note">Rows brighten as the clip plays.</span>', '<span class="rc-live-note">All rows shown; the clip that animates them is available on request.</span>')
    for cid in CLIPS:
        html = html.replace('{{VIDEO:%s}}' % cid, video_uri[cid]).replace('{{POSTER:%s}}' % cid, poster_uri[cid]).replace('{{STILL:%s}}' % cid, still_uri[cid])
    for sid in STILLS:
        html = html.replace('{{IMG:%s}}' % sid, img_uri[sid])
    html = re.sub(r'<!--/?clip(:[a-z0-9_]+)?-->', '', html)
    left = re.findall(r'\{\{[^}]{0,60}', html)
    assert not left, 'unfilled placeholders: %s' % left[:10]
    return html

full = fill(tpl, False); stills = fill(tpl, True)
out_full = OUT_DIR/'court-ledger.html'; out_stills = OUT_DIR/'court-ledger-stills.html'
for f, s in ((out_full, full), (out_stills, stills)):
    if f.exists(): f.unlink()
    f.write_text(s)
report['out'] = {'full': {'path': str(out_full), 'bytes': out_full.stat().st_size}, 'stills': {'path': str(out_stills), 'bytes': out_stills.stat().st_size}}
report['size_gate'] = {'total_le_12MB': out_full.stat().st_size <= 12_000_000, 'each_clip_le_1MB_b64': all(c['base64_bytes'] <= 1_000_000 for c in report['clips'].values())}
json.dump(report, open(SC/'assemble_report.json','w'), indent=1)
print(json.dumps(report, indent=1))
