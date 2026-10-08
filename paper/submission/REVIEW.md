# Review

The scientific sections are still placeholders. Grammar, consistency and claim checks of author-written sentences are waiting on that text. Nothing in Introduction, Methods, Implementation, Results, Discussion or the legends was rewritten.

What was checked on 8 October 2026:

- `scripts/check_submission_v100.py` writes `COMPLIANCE.md`. The remaining items there are author actions, not failed requirements, once the filler false alarm from the comment is ignored by the checker.
- `main.pdf` and `supplement.pdf` were rebuilt with tectonic. The placeholder main file is 4 pages. The budgeted filler, with Figures 2 and 6 in the supplement, was 6 pages.
- The 1200-dpi LZW TIFFs were rebuilt. They are gitignored.
- The numbers in the tables are macros from `paper/numbers.tex`.
- The supplement log records this check.

Not done, because there is no author prose yet:

- Sentence-level grammar.
- A claim-by-claim reading of the Results and Discussion.
- A check that every decimal in those sections is a macro. The checker will do the mechanical part of that when the text exists.
