# Review Q&A

**What is RAG?** Retrieval-augmented generation retrieves external passages
and supplies them to a language model before generation.

**What is machine unlearning?** It is the broader goal of removing the effect
of selected training data from a learned model or system.

**Why is vector deletion not weight unlearning?** It removes searchable
representations but leaves model parameters unchanged.

**Why compare hard deletion and prompt constraints?** They contrast a
retrieval-layer intervention with an instruction-only behavioral intervention.

**How is leakage measured?** This prototype checks expected lower-case
substrings in answers. It is transparent but not a semantic detector.

**What is utility?** Exact marker-match rate on unrelated control questions,
reported separately from target leakage.

**What is the contribution?** An auditable small protocol that separates
retrieval suppression, answer leakage, abstention, wrong answers, and retained
utility rather than collapsing them into one score.

**What are the baselines?** Baseline RAG, hard index deletion, and prompt
constraint.

**What stronger evidence is needed?** Larger independently authored tests,
semantic/manual leakage review, adversarial paraphrases, repeated seeds, and
an intervention that addresses model parameters or a formal unlearning
guarantee.

**What are the main limitations?** Synthetic data, tiny sample size, a small
CPU generator, lexical markers, and no claim about model-weight forgetting.
