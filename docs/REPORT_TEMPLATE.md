# Academic report template

## Abstract

State the problem, system, conditions, measured outcomes, and limitations in
150-250 words. Use only observed results.

## 1. Introduction

Motivation, RAG context, the research question, and scope.

## 2. Problem statement and objectives

Define target-fact removal, retained utility, and the distinction between
index deletion and model unlearning.

## 3. Related work

Discuss verified RAG and machine-unlearning sources. Mark any source awaiting
bibliographic verification.

## 4. Methodology

Describe the synthetic corpus, question types, conditions, controls,
procedure, seeds, models, and metric denominators.

## 5. System design and implementation

Describe ingestion, chunking, embeddings, FAISS, context construction,
generation, and rebuild-based deletion.

## 6. Experimental setup

Record Python version, package versions, model identifiers, hardware,
timestamped output directory, and exact commands.

## 7. Results

Insert the actual summary table and distinguish preserved pilot evidence from
new runs. Include target leakage, retrieval suppression, abstention,
unsupported/wrong answers, and control utility separately.

## 8. Discussion

Interpret what the results do and do not establish. Do not call index deletion
complete machine unlearning.

## 9. Limitations and threats to validity

Address synthetic data, small sample, lexical markers, generator behavior,
prompt sensitivity, and lack of weight-level guarantees.

## 10. Conclusion and future work

Summarize narrow findings and propose semantic review, larger datasets,
repeated seeds, adversarial tests, and stronger unlearning interventions.

## References

Include only verified bibliographic records with stable URLs or DOIs.
