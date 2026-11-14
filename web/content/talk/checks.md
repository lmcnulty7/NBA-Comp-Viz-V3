---
visual: none
---
# How the site keeps itself honest

- The public site shows a still frame. The clips sit only on the private copy, and the public build stops if a clip gets in.
- These slides exist only on the private copy, and the public build stops if they get in.
- Copy on screen is checked for reading level and for words a reader would not know.
- Browser tests open every page at phone and laptop width, and check that each value shown equals its data file.
- A model that failed, or found no signal, stays on the results page with its verdict.

## Notes

- npm run check runs, in order: generated types current, type check, lint, the data against the contract, the copy check (no digits, no long dashes, no banned phrases), the reading check, the contrast check, hygiene (no local paths, no email), unit tests, the public build, the no-video check, and Playwright on Chromium and WebKit at both widths.
- The reading grade is the Flesch-Kincaid formula, computed in the repository with no outside package, the same rule as the test of docs/SCRIPT.md.
- Every heavy job on the Mac takes one machine-wide lock, so builds, browser tests and model reruns never overlap.
