# Testing

## Executed

On 27 October 2026, the following command passed:

```powershell
python -m unittest discover -s tests -v
```

It ran 4 tests successfully. The tests cover chunk overlap, FAISS rebuild
deletion and post-deletion retrieval, refusal-versus-wrong-answer
classification, and summary denominators.

Pylance syntax checks also reported no syntax errors for
`src/run_benchmark.py` and `tests/test_core.py`.

The full benchmark command was attempted with the configured `.venv` and
failed before execution because FLAN-T5 model loading raised
`OSError: [Errno 1455] The paging file is too small for this operation`.

## Manual checks still required

After increasing Windows virtual memory or using a machine with sufficient
memory, run a timestamped full benchmark, inspect every generated answer for
semantic leakage, and run the interactive CLI and deletion demonstration.
Confirm the output metadata and compare raw rows with the summary.
