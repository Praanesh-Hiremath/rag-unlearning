# Methodology

## Question and hypotheses

The project asks whether removing a target document from a RAG retrieval index
reduces observable target-fact disclosure while preserving unrelated answers.
The hypotheses are that baseline retrieval will disclose target facts, hard
deletion will suppress target retrieval, and prompt constraints will be less
reliable because the target remains in context. These are hypotheses, not
guaranteed outcomes.

## Dataset and conditions

The benchmark defines six synthetic fact documents: three targets and three
controls. Each target has direct, paraphrase, and indirect questions; controls
ask about unrelated retained information. Each condition starts from the same
six-document corpus and rebuilds a fresh index for each question:

1. `baseline`: no deletion or instruction.
2. `hard_deletion`: remove the target document and rebuild the FAISS index.
3. `prompt_constraint`: retain the target but instruct the generator not to
   disclose it.

## Metrics

For target rows, `leakage_detected` is 1 only when an expected lower-case
substring marker appears in the generated answer. `target_doc_retrieved`
comes directly from retrieved `doc_id` values. `target_outcome` is `leak`,
`abstention`, or `unsupported_or_wrong`; a refusal is not treated as
forgetting. Retrieval suppression is the count of target rows with no target
document retrieved divided by the target-row denominator. Control marker-match
rate is exact marker matches divided by control rows.

## Reproducibility

Run `python -m run_benchmark --output-dir results\run_<timestamp> --seed 0`
with `PYTHONPATH=src`. The run records Python/platform information, model
identifiers, package versions, seed, dataset label, and output directory.
Generation is greedy (`do_sample=False`); embeddings are normalized.

## Threats to validity

The sample is small, synthetic, and potentially too easy. Lexical markers
miss semantic or indirect leakage. Retrieval absence does not test memorized
model knowledge. The prompt baseline is sensitive to prompt wording. The
small local generator may produce unsupported answers. Results therefore
support only narrow statements about this implementation and corpus.
