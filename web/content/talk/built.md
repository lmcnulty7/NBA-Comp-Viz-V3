---
visual: none
---
# How it was built

Python makes the data. The site only reads it.

- A tracking pipeline turns TV video into spots on the court. A scorecard checks them against the sensor data.
- The models were fitted once, on training games, and then locked. Games kept out of training were scored one time.
- An export step reads the locked results and writes the data files of the site.
- The site is a set of static pages. It reads those files and nothing else.

## Notes

- The site is a Next.js static export in TypeScript. Python produces data, the web only reads it, through one JSON contract written as pydantic models (export/schemas.py). The TypeScript types are generated from it.
- The models are frozen at the freeze tag. The held-out games were scored once per lane group and are never scored again; the possession review shows held-out games only, so every prediction there is out of sample.
- Nothing under export/ or web/ fits, tunes or calibrates. The one place that reruns frozen fitting code is tools/, on training games only, and its output is used only when it gives back the stored predictions.
