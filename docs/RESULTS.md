# Results

## Evidence status

`results/benchmark_raw.csv` and `results/benchmark_summary.csv` are preserved
pilot outputs already present in the repository. They contain 36 rows: 9
targets and 3 controls across each of 3 conditions. No fresh full benchmark
result is claimed here: the attempted run on 27 October 2026 failed during
FLAN-T5 loading with Windows error 1455 (paging file too small).

## Preserved pilot summary

| Condition | Target queries | Marker leaks | Leak rate | Explicit abstentions | Unsupported/wrong | No target retrieved | Suppression rate | Controls | Control matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 9 | 8 | 8/9 | 0 | 1 | 0 | 0/9 | 3 | 3/3 |
| hard_deletion | 9 | 0 | 0/9 | 0 | 9 | 9 | 9/9 | 3 | 3/3 |
| prompt_constraint | 9 | 7 | 7/9 | 0 | 2 | 0 | 0/9 | 3 | 3/3 |

These values were read from the preserved CSV and are marker-based
observations. Hard deletion's 9/9 retrieval suppression demonstrates that the
target chunks are absent from the rebuilt active index in those rows. It does
not demonstrate complete forgetting by the generator or embedding model.
The nine hard-deletion target answers were classified as unsupported/wrong,
not as valid abstentions.
