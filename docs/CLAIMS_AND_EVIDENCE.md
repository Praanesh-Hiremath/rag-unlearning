# Claims and evidence

| Proposed claim | Evidence | Strength | Limitation | Safe for paper? |
|---|---|---|---|---|
| Hard deletion removes target chunks from the active index | `VectorStore.delete_document`; 4-test suite; pilot `target_doc_retrieved=0` for 9/9 | Strong for this implementation | Does not alter model weights | Yes, narrowly |
| Hard deletion suppresses target retrieval in this pilot | `results/benchmark_summary.csv`, 9/9 | Moderate | Tiny synthetic test set | Yes, with scope |
| Hard deletion proves complete machine unlearning | No evidence | None | Retrieval deletion is not weight unlearning | No |
| Prompt constraints reliably prevent disclosure | Pilot shows 7/9 marker leaks | Contradicted by pilot | Lexical detector only | No |
| Unrelated utility was retained in the pilot | Controls matched 3/3 per condition | Moderate | Only three controls and exact markers | Yes, cautiously |
| Marker matching measures all leakage | No | None | Misses paraphrase and indirect disclosure | No |
| The benchmark is reproducible | Seed, model IDs, package metadata, and fixed synthetic definitions in code | Moderate | Fresh full run blocked in current environment | Yes, as a protocol |
| Results generalize to production RAG or real private data | No | None | Synthetic corpus and small local model | No |
