# Mission: DocSem train ambiguity audit

Produce a reproducible automated screen and independent agent text audit for every one of the 908 DocSem training records. Persist a label-blind question layer before benchmark comparison, use the exact three-axis taxonomy, keep semantic and benchmark answers separate, preserve evidence provenance, and identify records requiring human review.

Success requires:

- 908 output rows and 908 unique source IDs, with no missing or extra IDs;
- 908 blind rows whose hashes are consumed by the benchmark-comparison pass;
- exact axis and severity values from the Issue 14 contract;
- 908 agent text-audit decisions over exact A/B/C boundaries;
- 273 genuine reviewer overrides and 635 explicit clean candidates;
- zero PDF visual-review or human-adjudication claims;
- a passing stdlib validator artifact at `result.json`.
