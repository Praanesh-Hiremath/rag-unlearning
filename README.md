# RAG Machine Unlearning

This repository is an undergraduate research prototype for studying whether
information can be selectively removed from a retrieval-augmented generation
(RAG) system. The research question is: **after a target document is removed
from retrieval, does the system still disclose the target fact, and is
unrelated utility retained?**

The project compares a baseline RAG system, hard deletion from a FAISS index,
and a prompt-constraint baseline. The default corpus is synthetic and
contains no private or copyrighted source material.

## Features

- Text ingestion, word-based chunking, normalized Sentence Transformer
  embeddings, FAISS inner-product retrieval, and local FLAN-T5 generation.
- Auditable document IDs and retrieval passages.
- Rebuild-based deletion that removes every chunk belonging to a document.
- Deterministic, marker-based benchmark metrics with explicit denominators.
- Unit tests that run without downloading or loading the generation model.

## Requirements and installation

Windows PowerShell and Python 3.12 were used for the current environment.
CPU execution is supported; a GPU, paid API, Docker, and external database are
not required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first application or benchmark run downloads the configured Hugging Face
models. Do not commit the resulting model cache.

## Run the application

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python src\main.py
```

Ask a question when prompted. The CLI prints retrieved document IDs, scores,
passages, and the generated answer. The more focused deletion demonstration
is:

```powershell
$env:PYTHONPATH = "src"
python src\demo.py
```

## Run tests and benchmark

```powershell
python -m unittest discover -s tests -v
$env:PYTHONPATH = "src"
python -m run_benchmark --output-dir results\run_YYYYMMDD_HHMMSS --seed 0
```

Use a unique timestamped output directory to preserve evidence. The benchmark
writes `benchmark_raw.csv`, `benchmark_summary.csv`, and `run_metadata.json`.
The checked-in `results\benchmark_*.csv` files are preserved pilot evidence,
not an automatically verified fresh run.

## Metrics and interpretation

Target rows record exact marker-detected leakage, target-document retrieval,
explicit abstention phrases, and `unsupported_or_wrong` answers. Control rows
record exact marker matches for unrelated questions. Marker matching is a
transparent lexical check, not a semantic leakage detector. Retrieval
suppression is evidence about the active index, not proof that model weights
forgot the fact.

The preserved pilot summary reports 9 target and 3 control queries per
condition: baseline 8/9 marker leaks, hard deletion 0/9, and prompt constraint
7/9; all conditions have 3/3 control marker matches. These figures must be
described as marker-based synthetic pilot observations. A fresh run could not
be completed in the current environment because loading FLAN-T5 failed with
Windows error 1455 (paging file too small).

## Repository guide

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - implementation pipeline.
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) - protocol and threats to validity.
- [docs/RESULTS.md](docs/RESULTS.md) - preserved evidence and interpretation.
- [docs/TESTING.md](docs/TESTING.md) - executed checks and limitations.
- [docs/CLAIMS_AND_EVIDENCE.md](docs/CLAIMS_AND_EVIDENCE.md) - safe claims.
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) - five-to-seven-minute demo.
- [docs/REVIEW_QA.md](docs/REVIEW_QA.md) - likely examiner questions.
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md) - sources and verification status.
- [docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md) - academic report structure.

## Known limitations

The corpus is tiny and synthetic, the generator is small, and the benchmark
uses three conditions with few questions. Exact markers miss paraphrases,
synonyms, and indirect disclosure. Hard deletion rebuilds the retrieval index;
it does not modify model weights or prove complete machine unlearning. Manual
review and a larger, independently authored evaluation set are required for
stronger claims.
