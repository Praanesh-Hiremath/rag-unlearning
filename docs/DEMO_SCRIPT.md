# Five-to-seven-minute demonstration

1. **0:00-0:45 — Introduce the question.** From the repository root, say
   that the system tests retrieval/index deletion, not model-weight
   unlearning.
2. **0:45-2:00 — Run the CLI.**
   ` $env:PYTHONPATH="src"; python src\main.py ` and ask about the
   laboratory. Show document IDs, scores, context, and answer.
3. **2:00-3:30 — Show deletion.** Run
   ` $env:PYTHONPATH="src"; python src\demo.py `. Point out the target
   document ID before deletion, the rebuilt vector count afterward, and the
   changed retrieval set.
4. **3:30-4:45 — Explain prompt constraint.** Show that the target remains in
   the index while the instruction attempts to suppress disclosure. Explain
   why this is a behavioral baseline, not forgetting.
5. **4:45-6:00 — Show evidence.** Open
   `results\benchmark_summary.csv` and explain the three denominators and
   separate metrics. Mention that the preserved pilot has 8/9 baseline,
   0/9 hard-deletion, and 7/9 prompt-constraint marker leaks.
6. **6:00-7:00 — State limitations.** Exact markers are incomplete; wrong
   answers are not abstentions; index deletion is not model unlearning. If a
   fresh run differs, show its timestamped directory and report it instead.
