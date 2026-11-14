---
visual: none
---
# One set of data files, one set of rules

Every number on the site comes from a data file. Each file names the reports and tables it was read from.

- A rule file says what each data file must hold. The export and the site both check every file against it.
- The page copy holds words only. A number in the copy is a named field of the data, filled in when the site is built.
- A field name the build does not know stops the build.
- The same files could later come from a live service, and the pages would not change.

## Notes

- The contract is export/schemas.py. It emits JSON Schema, the site checks its data with ajv and then with the pydantic validators.
- The cold open copy is docs/SCRIPT.md. Its numbers are tokens such as the fill share token, and the token table says which field each one reads and how it may be shown: a share moved two decimal places to read as a percentage, never rounded.
- The live service is the next step after this version: the same shapes from a feed, so the review page can become a bench tool.
