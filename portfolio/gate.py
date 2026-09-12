#!/opt/anaconda3/bin/python
"""Grep gate from brief §E step 5. Prints every hit with context; exit 1 on any hit."""
import re, sys, html as H
from pathlib import Path
path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / 'build' / 'court-ledger.html')
raw = Path(path).read_text()
src = re.sub(r'data:[a-z/]+;base64,[A-Za-z0-9+/=]+', 'DATAURI', raw)   # never grep base64
src = re.sub(r'<script>.*?</script>', '', src, flags=re.S)
src = re.sub(r'<style>.*?</style>', '', src, flags=re.S)
# text with <s>...</s> (strikethrough) regions marked so allowed cells can be excluded
struck = [H.unescape(re.sub(r'<[^>]+>', '', m)) for m in re.findall(r'<s>(.*?)</s>', src, flags=re.S)]
text_keep_s = H.unescape(re.sub(r'<[^>]+>', ' ', src))
text_no_s = H.unescape(re.sub(r'<[^>]+>', ' ', re.sub(r'<s>.*?</s>', ' [STRUCK] ', src, flags=re.S)))
hits = []
def ctx(t, i, j, n=70): return t[max(0,i-n):j+n].replace('\n',' ')
def find_all(pat, t, flags=0):
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(pat, t, flags)]
def ban(pat, t=None, label=None, flags=re.I, where=None):
    t = text_no_s if t is None else t
    for i, j, g in find_all(pat, t, flags):
        if where and not where(t, i, j): continue
        hits.append((label or pat, ctx(t, i, j)))

# 91.1 outside the single struck-through ledger cell(s)
n_struck_911 = sum(s.count('91.1') for s in struck)
ban(r'91\.1', label='91.1 outside strikethrough')
if n_struck_911 == 0: hits.append(('91.1 must appear struck through exactly once', 'none found'))
ban(r'league average'); ban(r'above average'); ban(r'below average')
ban(r'matchup accuracy', where=lambda t,i,j: not re.search(r'(no |unmeasured)\s*$', t[max(0,i-12):i], re.I), label='matchup accuracy without no/unmeasured')
ban(r'identifies players'); ban(r'tracks each player'); ban(r'fully reproducible')
# byte-identical needs Colab or cached anchors in the same sentence
for i, j, g in find_all(r'byte-identical', text_no_s, re.I):
    a = text_no_s.rfind('.', 0, i) + 1; b = text_no_s.find('.', j); b = len(text_no_s) if b < 0 else b
    sent = text_no_s[a:b]
    if 'Colab' not in sent and 'cached anchors' not in sent: hits.append(('byte-identical without Colab/cached anchors', ctx(text_no_s, i, j)))
for w in ('elite', 'rated', 'ranked', 'representative', 'unbiased'):
    ban(r'\b%s\b' % w, label='word: '+w)
ban(r'home team'); ban(r'away team')
ban(r'home/away', where=lambda t,i,j: not (re.search(r'not\s*$', t[max(0,i-6):i], re.I) or '≠' in t[max(0,i-12):j+12]), label='home/away without ≠ / not')
for w in ('transition defense', 'help defense', 'shot quality', 'closeout rating'):
    ban(re.escape(w), label=w+' outside kill list')
ban(r'defensive value'); ban(r'auto-label accuracy 95')
ban(r'40\.3\s*%', where=lambda t,i,j: 'not a meaningful number' not in t[i:j+140], label='40.3% without its sentence')
ban(r'279', where=lambda t,i,j: re.search(r'\bft\b', t[max(0,i-40):j+40]) is not None, label='279 within 40 chars of ft')
ban(r'half the dataset'); ban(r'half of')
ban(r'10\.58'); ban(r'Every Frame', flags=0)
for i, j, g in find_all(r'\b(Curry|Durant|West|Jones|Livingston|Thompson)\b', text_no_s):
    win = text_no_s[max(0,i-80):j+80]
    if re.search(r'credit|per 100', win, re.I): hits.append(('surname near credit/per 100: '+g, ctx(text_no_s, i, j, 90)))
# informational: literal substring 'ft' within 40 chars of 279 (stricter reading)
info = [ctx(text_no_s,i,j) for i,j,g in find_all(r'279', text_no_s) if 'ft' in text_no_s[max(0,i-40):j+40]]
print('GREP GATE on', path)
for lab, c in hits: print('HIT [%s]: …%s…' % (lab, c))
print('hits:', len(hits), '| 91.1 struck-through occurrences:', n_struck_911, '| informational: literal "ft" substring within 40 chars of 279:', len(info))
for c in info: print('  info:', c)
sys.exit(1 if hits else 0)
