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

# ---- charts: HTML labels, SVG marks (text never scales with a viewBox) --------
def _pct(v, lo, hi, p0=0.0, p1=100.0):
    return p0 + (v - lo) / (hi - lo) * (p1 - p0)

def chart_tile_xval():
    """Hero tile 1: one dot per game on a 0.70–1.00 axis, pooled rate as a line."""
    pg = N['crossval_per_game']['per_game']; pooled = N['crossval']['rate']
    lo, hi = 0.70, 1.00
    X = lambda v: '%.1f%%' % _pct(v, lo, hi, 2, 98)
    dots = ''.join('<circle class="dot-g" cx="%s" cy="14" r="3.4"><title>%s · %.1f%% (n=%d)</title></circle>'
                   % (X(g['agreement_rate']), k, g['agreement_rate']*100, g['n_checked']) for k, g in sorted(pg.items()))
    return ('<svg viewBox="0 0 320 34" preserveAspectRatio="none" height="34" aria-hidden="true" style="width:100%">'
            '<line class="ax" x1="2%" x2="98%" y1="14" y2="14"/>'
            + ''.join('<line class="ax" x1="%s" x2="%s" y1="10" y2="18"/>' % (X(t), X(t)) for t in (0.7, 0.8, 0.9, 1.0))
            + '<line class="ref" x1="%s" x2="%s" y1="3" y2="25"/>' % (X(pooled), X(pooled))
            + dots
            + '<text x="2%" y="32">70%</text><text x="98%" y="32" text-anchor="end">100%</text>'
            + '<text x="%s" y="32" text-anchor="middle">pooled %.1f%%</text>' % (X(pooled), pooled*100)
            + '</svg>')

def chart_tile_ci():
    """Hero tile 2: the interval itself, zero marked."""
    g = N['credit']['gsw']; lo, hi = g['ci95']; pt = g['credit_per_100']
    x0, x1 = -30.0, 15.0
    X = lambda v: '%.1f%%' % _pct(v, x0, x1, 3, 97)
    out = '<svg viewBox="0 0 320 34" preserveAspectRatio="none" height="34" aria-hidden="true" style="width:100%">'
    out += '<line class="ax" x1="3%" x2="97%" y1="14" y2="14"/>'
    out += '<line class="zero" x1="%s" x2="%s" y1="3" y2="25"/>' % (X(0), X(0))
    out += '<line class="bar-h" x1="%s" x2="%s" y1="14" y2="14"/>' % (X(lo), X(hi))
    out += '<circle class="dot-h" cx="%s" cy="14" r="4.5"/>' % X(pt)
    out += '<text x="%s" y="32" text-anchor="middle">0</text>' % X(0)
    out += '<text x="%s" y="32" text-anchor="end">%s</text>' % (X(lo), minus('%.1f' % lo))
    out += '<text x="%s" y="32">+%.1f</text>' % (X(hi), hi)
    return out + '</svg>'

def chart_tile_unm():
    """Hero tile 3: an axis with nothing on it — a dashed placeholder where a point would be."""
    return ('<svg viewBox="0 0 320 34" preserveAspectRatio="none" height="34" aria-hidden="true" style="width:100%">'
            '<line class="ax" x1="3%" x2="97%" y1="14" y2="14" stroke-dasharray="3 3"/>'
            '<rect class="ghost" x="46%" y="7" width="8%" height="14"/>'
            '<text x="3%" y="32">0%</text><text x="97%" y="32" text-anchor="end">100%</text>'
            '<text x="50%" y="32" text-anchor="middle">no measurement</text>'
            '</svg>')

def chart_rail():
    """The stage rail as a spec strip: each node carries its headline score as a bar."""
    g, d, h, t, po, c = N['gate'], N['detection'], N['homography'], N['teams'], N['possessions'], N['clock']
    f1 = float(d['f1_str'])
    nodes = [
      ('m', '#s1', '1', 'GATE', '%s%% · %d held-out' % (g['accuracy_pct'], g['n_test']), g['accuracy_pct']),
      ('m', '#s2', '2', 'DETECT + TRACK', 'F1 %s · %d boxes' % (d['f1_str'], d['total_gt_boxes']), f1*100),
      ('m', '#f-s2', '3', 'COURT', '%s ft · %d frames' % (h['median_ft_str'], h['val_frames']), h['homography_success_rate']*100),
      ('h', '#f-s6', '4', 'IDENTITY', 'churn 6.57→5.0 · label-free', None),
      ('m', '#s3', '5', 'TEAMS', '%s%% · %d labels' % (t['track_level_accuracy_pct'], t['n_labeled']), t['track_level_accuracy_pct']),
      ('m', '#s3', '6', 'POSSESSIONS', '%s / %s%% · %d of %d' % (po['basket_accuracy_pct'], po['offense_accuracy_pct'], po['basket_n'], po['n_labeled']), po['basket_accuracy_pct']),
      ('u', '#s4', '7', 'MATCHUPS', 'UNMEASURED · no number', None),
      ('m', '#s3', '8', 'CLOCK + PBP', '%d%% on readable · %d labels' % (round(c['accuracy_on_readable_pct']), c['n']), c['accuracy_on_readable_pct']),
      ('j', '#s5', '→', 'JOIN → CREDIT', '%d possessions · an interval' % N['credit']['joined_total'], None),
    ]
    out = ['<div class="rail" id="rail">']
    for cls, href, k, name, ro, score in nodes:
        bar = '' if score is None and cls != 'u' else ('<div class="bar"></div>' if score is None else '<div class="bar"><i style="width:%.1f%%"></i></div>' % score)
        tip = ' data-tip="%s"' % esc(ro)
        out.append('<a class="node %s" href="%s" aria-label="Stage %s %s — %s — jump"%s><span class="k">%s</span><span class="name">%s</span><span class="ro">%s</span>%s</a>'
                   % (cls, href, esc(k), esc(name), esc(ro), tip, esc(k), esc(name), esc(ro), bar))
    out.append('</div>')
    return ''.join(out)

def chart_gate():
    """Fig. 1: three bars — trained head vs the two baselines, FN/FP labelled."""
    g = N['gate']
    rows = [('trained probe (this work)', g['accuracy_pct'], '%d FN / %d FP' % (g['fn'], g['fp']), 'f-m'),
            ('CLIP zero-shot', g['zero_shot_clip_accuracy_pct'], '15 FN / 3 FP · drops live frames', 'f-pop'),
            ('HSV floor-color rule', g['hsv_baseline_accuracy_pct'], '0 FN / 20 FP · lets dead frames through', 'f-pop')]
    X = lambda v: '%.1f%%' % _pct(v, 80, 100, 0, 100)
    out = ['<div class="rows" role="img" aria-label="Gate accuracy: trained probe %s%%, CLIP zero-shot %s%%, HSV rule %s%%, on %d held-out frames">' % (g['accuracy_pct'], g['zero_shot_clip_accuracy_pct'], g['hsv_baseline_accuracy_pct'], g['n_test'])]
    out.append('<span></span><div class="ticks">' + ''.join('<span style="left:%s">%d%%</span>' % (X(t), t) for t in (80, 85, 90, 95, 100)) + '</div><span></span>')
    for lab, v, note, cls in rows:
        out.append('<span class="lab">%s</span>' % esc(lab))
        out.append('<svg height="20" aria-hidden="true" class="row-hi" data-tip="%s · %s%% · %s">' % (esc(lab), v, esc(note))
                   + ''.join('<line class="ax" x1="%s" x2="%s" y1="0" y2="20"/>' % (X(t), X(t)) for t in (80, 85, 90, 95, 100))
                   + '<rect class="%s" x="0" y="3" width="%s" height="14"/>' % (cls, X(v))
                   + '</svg>')
        out.append('<span class="val">%s%% · %s</span>' % (v, esc(note)))
    out.append('</div>')
    return ''.join(out)

def chart_det():
    """Fig. 2: dumbbell — P, R, F1 before → after the basketball retrain."""
    d = N['detection']
    before = {'precision': 0.68, 'recall': 0.74, 'F1': 0.71}   # hand-typed source: detection_baseline_f1_0_71_p_0_68_r_0_74
    after = {'precision': float(d['precision_str']), 'recall': float(d['recall_str']), 'F1': float(d['f1_str'])}
    X = lambda v: '%.1f%%' % _pct(v, 0.6, 1.0, 0, 100)
    out = ['<div class="rows" role="img" aria-label="Detection before and after fine-tuning: precision 0.68 to %s, recall 0.74 to %s, F1 0.71 to %s">' % (d['precision_str'], d['recall_str'], d['f1_str'])]
    out.append('<span></span><div class="ticks">' + ''.join('<span style="left:%s">%.1f</span>' % (X(t), t) for t in (0.6, 0.7, 0.8, 0.9, 1.0)) + '</div><span></span>')
    for k in ('precision', 'recall', 'F1'):
        b, a = before[k], after[k]
        out.append('<span class="lab">%s</span>' % k)
        out.append('<svg height="20" aria-hidden="true" class="row-hi" data-tip="%s: %.2f → %.2f">' % (k, b, a)
                   + ''.join('<line class="ax" x1="%s" x2="%s" y1="0" y2="20"/>' % (X(t), X(t)) for t in (0.6, 0.7, 0.8, 0.9, 1.0))
                   + '<line class="bar-m" x1="%s" x2="%s" y1="10" y2="10"/>' % (X(b), X(a))
                   + '<circle class="dot-x" cx="%s" cy="10" r="4.5" style="opacity:.55"/>' % X(b)
                   + '<circle class="dot-m" cx="%s" cy="10" r="4.5"/>' % X(a)
                   + '</svg>')
        out.append('<span class="val">%.2f → <b>%.2f</b></span>' % (b, a))
    out.append('</div>')
    out.append('<div class="legend"><span><i style="background:var(--excluded);opacity:.55;border-radius:50%"></i>COCO person, off the shelf</span><span><i style="background:var(--measured);border-radius:50%"></i>YOLOv8m fine-tuned on basketball</span></div>')
    return ''.join(out)

def chart_cov():
    """Fig. 3: matchup coverage — p10/median/p90 across the corpus, with the two shown possessions marked."""
    m = N['matchup_coverage_corpus']; s729 = N['matchup_span_729']['coverage_pairs_per_frame']
    q1 = N['matchup_curry_q1_span_0']
    X = lambda v: '%.1f%%' % _pct(v, 0, 5, 2, 98)
    p10, med, p90 = m['non_degraded_p10'], m['non_degraded_median'], m['non_degraded_p90']
    return ('<div class="callout-num">%.1f <small style="font-family:var(--mono);font-size:13px;font-weight:400;color:var(--muted)">pairs per frame, corpus median</small></div>' % med
        + '<svg viewBox="0 0 640 64" preserveAspectRatio="none" height="64" role="img" aria-label="Matchup coverage across %d non-degraded possessions: p10 %.1f, median %.1f, p90 %.1f pairs per frame; the shown possession draws %.1f" style="width:100%%">' % (m['n_non_degraded'], p10, med, p90, s729)
        + '<rect class="f-soft" x="%s" y="18" width="%.1f%%" height="16"><title>p10–p90 band</title></rect>' % (X(p10), _pct(p90, 0, 5, 2, 98) - _pct(p10, 0, 5, 2, 98))
        + '<line class="ax" x1="2%" x2="98%" y1="26" y2="26"/>'
        + ''.join('<line class="ax" x1="%s" x2="%s" y1="22" y2="30"/><text class="mu" x="%s" y="46" text-anchor="middle">%d</text>' % (X(t), X(t), X(t), t) for t in range(0, 6))
        + '<line class="zero" x1="%s" x2="%s" y1="12" y2="40"/>' % (X(med), X(med))
        + '<circle class="dot-m" cx="%s" cy="26" r="5" data-tip="gsw_hou_duel_s01 span 729: %.1f pairs/frame, 0%% excluded"/>' % (X(s729), s729)
        + '<text x="%s" y="10" text-anchor="middle">this possession %.1f</text>' % (X(s729), s729)
        + '<text class="mu" x="%s" y="60" text-anchor="middle">p10 %.1f</text><text class="mu" x="%s" y="60" text-anchor="middle">p90 %.1f</text>' % (X(p10), p10, X(p90), p90)
        + '<text class="mu" x="98%" y="60" text-anchor="end">pairs / frame</text>'
        + '</svg>'
        + '<div class="hint">n = %s non-degraded possessions · the second clip\'s possession excludes %d of %d sampled frames outright (%s%%) rather than pairing them</div>' % (format(m['n_non_degraded'], ','), q1['frames_excluded_team_gt5'], q1['frames_used'] + q1['frames_excluded_team_gt5'], q1['pct_excluded']))

def chart_f1():
    """Fig. 4: the interval, full width — HTML title row, SVG marks only."""
    g = N['credit']['gsw']; lo, hi = g['ci95']; pt = g['credit_per_100']
    x0, x1 = -30.0, 15.0
    X = lambda v: '%.2f%%' % _pct(v, x0, x1, 4, 96)
    return ('<div class="rows" style="grid-template-columns:auto 1fr auto">'
        '<span class="lab"><i class="g g-h"></i>GSW · n = %d · %d games</span>' % (g['n_possessions'], g['n_games'])
        + '<svg viewBox="0 0 640 56" preserveAspectRatio="none" height="56" role="img" aria-label="Interval chart: GSW credit per 100 possessions, point %s, 95%% CI %s to %s" style="width:100%%">' % (minus(str(pt)), minus(str(lo)), minus(str(hi)))
        + '<line class="ax" x1="4%" x2="96%" y1="28" y2="28"/>'
        + ''.join('<line class="ax" x1="%s" x2="%s" y1="24" y2="32"/>' % (X(t), X(t)) for t in (-30, -20, -10, 0, 10))
        + '<line class="zero" x1="%s" x2="%s" y1="6" y2="50"/>' % (X(0), X(0))
        + '<line class="bar-h" x1="%s" x2="%s" y1="28" y2="28" data-tip="95%% CI %s to +%s"/>' % (X(lo), X(hi), minus(str(lo)), hi)
        + '<circle class="dot-h" cx="%s" cy="28" r="6" data-tip="point estimate %s per 100"/>' % (X(pt), minus(str(pt)))
        + '</svg>'
        + '<span class="val">%s [%s, +%s]</span></div>' % (minus('%.1f' % pt), minus('%.1f' % lo), hi)
        + '<div class="rows" style="grid-template-columns:auto 1fr auto"><span></span><div class="ticks">'
        + ''.join('<span style="left:%s">%s</span>' % (X(t), ('zero' if t == 0 else minus(str(t)))) for t in (-30, -20, -10, 0, 10))
        + '</div><span></span></div>'
        + '<div class="hint">credit per 100 possessions → · negative = offenses scored more against this defense than their own context-matched norm</div>')

def chart_xval_games():
    """Fig. 5: per-game lollipop — dot at agreement, bar length ∝ possessions checked, pooled line."""
    pg = N['crossval_per_game']['per_game']; pooled = N['crossval']['rate']
    items = sorted(pg.items(), key=lambda kv: kv[1]['agreement_rate'])
    nmax = max(g['n_checked'] for _, g in items)
    lo, hi = 0.70, 1.00
    X = lambda v: '%.1f%%' % _pct(v, lo, hi, 0, 100)
    out = ['<div class="rows" role="img" aria-label="Per-game agreement with play-by-play, %d games, range %s, pooled %.1f%%">' % (len(items), N['crossval_per_game']['range_str'], pooled*100)]
    out.append('<span></span><div class="ticks">' + ''.join('<span style="left:%s">%d%%</span>' % (X(t), round(t*100)) for t in (0.7, 0.8, 0.9, 1.0)) + '</div><span></span>')
    for k, g in items:
        r = g['agreement_rate']; n = g['n_checked']
        label = '%s-%s-%s %s' % (k[:4], k[4:6], k[6:8], k[9:])
        out.append('<span class="lab">%s</span>' % esc(label))
        out.append('<svg height="18" aria-hidden="true" class="row-hi" data-tip="%s · %d of %d agree (%.1f%%)">' % (esc(label), g['agree'], n, r*100)
                   + ''.join('<line class="ax" x1="%s" x2="%s" y1="0" y2="18"/>' % (X(t), X(t)) for t in (0.7, 0.8, 0.9, 1.0))
                   + '<line class="ref" x1="%s" x2="%s" y1="0" y2="18"/>' % (X(pooled), X(pooled))
                   + '<rect class="f-mu" x="0" y="6" width="%s" height="6" style="opacity:%.2f"/>' % (X(lo + (hi-lo) * n / nmax), 0.25)
                   + '<line class="bar-m" x1="%s" x2="%s" y1="9" y2="9" style="stroke-width:1.5;opacity:.5"/>' % (X(lo), X(r))
                   + '<circle class="%s" cx="%s" cy="9" r="4.5"/>' % ('dot-m' if r >= pooled else 'dot-h', X(r))
                   + '</svg>')
        out.append('<span class="val">%.1f%% · n=%d</span>' % (r*100, n))
    out.append('</div>')
    out.append('<div class="legend"><span><i style="background:var(--measured);border-radius:50%%"></i>at or above the pooled rate</span><span><i style="background:var(--held);border-radius:50%%"></i>below it</span><span><i style="background:var(--muted);opacity:.35"></i>bar = possessions checked (max %d)</span></div>' % nmax)
    return ''.join(out)

def chart_funnel():
    """Fig. 6: three real populations as proportional bars + the anchor-failure breakdown."""
    b = N['bias_audit']; c = N['crossval']
    steps = [('all possessions in the %d games' % b['games'], b['population_all']['n'], 'f-pop'),
             ('halfcourt subset (the pipeline\'s target)', b['population_halfcourt']['n'], 'f-pop'),
             ('joined into the credit table', b['sampled_joined']['n'], 'f-m')]
    top = steps[0][1]
    out = ['<div class="rows" role="img" aria-label="Funnel: %s all, %s halfcourt, %s joined possessions" style="grid-template-columns:1fr auto">' % tuple(format(s[1], ',') for s in steps)]
    for lab, n, cls in steps:
        w = 100.0 * n / top
        out.append('<div style="display:grid;gap:3px"><span class="lab">%s</span><svg height="18" aria-hidden="true" class="row-hi" data-tip="%s: %s (%.0f%% of all)"><rect class="%s" x="0" y="0" width="%.1f%%" height="18"/></svg></div>' % (esc(lab), esc(lab), format(n, ','), w, cls, w))
        out.append('<span class="val" style="align-self:end">%s</span>' % format(n, ','))
    out.append('</div>')
    fr = c['failure_reasons']; tot = c['anchor_failures']; aligned = c['possessions_aligned']
    parts = [('anchor_failed', fr['anchor_failed'], 'f-x'), ('no_pbp_overlap', fr['no_pbp_overlap'], 'f-h'), ('clock_stopped_span', fr['clock_stopped_span'], 'f-mu'), ('anchor_inconsistent', fr['anchor_inconsistent'], 'f-pop')]
    total = aligned + tot
    x = 0.0
    seg = ''
    for name, n, cls in parts:
        w = 100.0 * n / total
        seg += '<rect class="%s" x="%.2f%%" y="0" width="%.2f%%" height="14" data-tip="%s: %d"><title>%s: %d</title></rect>' % (cls, x, max(w - 0.3, 0), name, n, name, n)
        x += w
    out.append('<div class="hint" style="margin-top:6px">of the spans that reached alignment: <b style="color:var(--ink)">%s aligned</b> · %s failed anchoring —</div>' % (format(aligned, ','), tot))
    out.append('<svg viewBox="0 0 640 14" preserveAspectRatio="none" height="14" aria-hidden="true" style="width:100%%"><rect class="f-m" x="0" y="0" width="%.2f%%" height="14" data-tip="aligned: %d"/>%s</svg>' % (100.0 * aligned / total, aligned, seg))
    out.append('<div class="legend"><span><i style="background:var(--measured)"></i>aligned %d</span>' % aligned + ''.join('<span><i class="%s" style="background:%s"></i>%s %d</span>' % (cls, {'f-x':'var(--excluded)','f-h':'var(--held)','f-mu':'var(--muted)','f-pop':'var(--pop)'}[cls], name.replace('_', ' '), n) for name, n, cls in parts) + '</div>')
    return ''.join(out)

def chart_mix():
    """Fig. 7: points-mix paired bars — halfcourt population vs joined sample."""
    b = N['bias_audit']
    pop = b['population_halfcourt']['points_mix']; joi = b['sampled_joined']['points_mix']
    cats = ['0', '1', '2', '3+']
    X = lambda v: '%.1f%%' % _pct(v, 0, 0.6, 0, 100)
    out = ['<div class="rows" role="img" aria-label="Points per possession mix, halfcourt population vs joined sample: %s">' % '; '.join('%s pts: %.1f%% vs %.1f%%' % (c, pop[c]*100, joi[c]*100) for c in cats)]
    out.append('<span></span><div class="ticks">' + ''.join('<span style="left:%s">%d%%</span>' % (X(t), round(t*100)) for t in (0, .2, .4, .6)) + '</div><span></span>')
    for c in cats:
        pv, jv = pop[c], joi[c]
        out.append('<span class="lab">%s pts</span>' % c)
        out.append('<svg height="26" aria-hidden="true" class="row-hi" data-tip="%s points: population %.1f%% → joined %.1f%%">' % (c, pv*100, jv*100)
                   + ''.join('<line class="ax" x1="%s" x2="%s" y1="0" y2="26"/>' % (X(t), X(t)) for t in (0, .2, .4, .6))
                   + '<rect class="f-pop" x="0" y="3" width="%s" height="9"/>' % X(pv)
                   + '<rect class="f-t" x="0" y="14" width="%s" height="9"/>' % X(jv)
                   + '</svg>')
        out.append('<span class="val">%.1f%% → <b>%.1f%%</b></span>' % (pv*100, jv*100))
    out.append('</div>')
    out.append('<div class="legend"><span><i style="background:var(--pop);opacity:.55"></i>halfcourt population (n = %s)</span><span><i style="background:var(--template)"></i>joined sample (n = %s)</span></div>' % (format(b['population_halfcourt']['n'], ','), format(b['sampled_joined']['n'], ',')))
    return ''.join(out)

def chart_f5():
    """Fig. 8: the audit — dot at accuracy, Wilson bar, dashed aggregate line; rows sorted."""
    rows = N['label_audit']['f5_chart_rows']; agg = N['label_audit']['overall']['accuracy']
    X = lambda v: '%.2f%%' % (1.0 + v*98.0)
    out = ['<div class="rows" role="img" aria-label="Dot-and-interval chart of audit accuracy by stratum with Wilson 95%% intervals; aggregate %.1f%% is not a meaningful number because it mixes bands">' % (agg*100)]
    out.append('<span></span><div class="ticks">' + ''.join('<span style="left:%s">%d%%</span>' % (X(t), round(t*100)) for t in (0, .25, .5, .75, 1)) + '</div><span></span>')
    for r in rows:
        lo, hi = r['wilson95']; m = r['band'] == 'agreement'
        cls_bar = 'bar-m' if m else 'bar-x'; cls_dot = 'dot-m' if m else 'dot-x'
        out.append('<span class="lab"><i class="g %s"></i>%s</span>' % ('g-m' if m else 'g-u', esc(r['stratum'])))
        out.append('<svg height="24" aria-hidden="true" class="row-hi" data-tip="%s · %.1f%% [%.1f, %.1f] · n=%d">' % (esc(r['stratum']), r['accuracy']*100, lo*100, hi*100, r['n_judged'])
            + ''.join('<line class="ax" x1="%s" x2="%s" y1="0" y2="24"/>' % (X(t), X(t)) for t in (0,.25,.5,.75,1))
            + '<line class="ref" x1="%s" x2="%s" y1="0" y2="24"/>' % (X(agg), X(agg))
            + '<line class="%s" x1="%s" x2="%s" y1="12" y2="12"/>' % (cls_bar, X(lo), X(hi))
            + '<circle class="%s" cx="%s" cy="12" r="5"/>' % (cls_dot, X(r['accuracy']))
            + '</svg>')
        out.append('<span class="val">%.1f%% · n=%d</span>' % (r['accuracy']*100, r['n_judged']))
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

CHARTS = {'TILE_XVAL': chart_tile_xval, 'TILE_CI': chart_tile_ci, 'TILE_UNM': chart_tile_unm, 'RAIL': chart_rail,
          'GATE': chart_gate, 'DET': chart_det, 'COV': chart_cov, 'F1': chart_f1, 'XVAL_GAMES': chart_xval_games,
          'FUNNEL': chart_funnel, 'MIX': chart_mix, 'F5': chart_f5, 'NUMLINE': chart_numline}

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
    html = html.replace('{{STYLE}}', (SC / 'style.css').read_text())
    html = re.sub(r'\{\{CHART:([A-Z0-9_]+)\}\}', lambda mo: CHARTS[mo.group(1)](), html)
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
