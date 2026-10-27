from pathlib import Path

from ingestion import load_text_documents
from chunking import chunk_documents
from embeddings import Embedder
from vector_store import VectorStore
from generator import Generator


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "original"


def build_store(documents, embedder):

    if not documents:
        raise ValueError(
            f"No documents were loaded from {DATA_DIR}. "
            "Check that the directory exists and contains supported .txt files."
        )

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        overlap=20
    )

    if not chunks:
        raise ValueError(
            f"No chunks were created from documents in {DATA_DIR}. "
            "Check that the input documents contain text."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.encode(texts)

    if getattr(embeddings, "ndim", None) != 2:
        raise ValueError(
            "Embedding output must be a two-dimensional array with one "
            f"row per chunk; received shape {getattr(embeddings, 'shape', None)}."
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Embedding row count must match chunk count: "
            f"{embeddings.shape[0]} embeddings for {len(chunks)} chunks."
        )

    if embeddings.shape[1] == 0:
        raise ValueError("Embedding output has zero dimensions.")

    store = VectorStore(
        embeddings.shape[1]
    )

    store.add(
        embeddings,
        chunks
    )

    return store


def retrieve(
    store,
    embedder,
    question
):

    query_embedding = embedder.encode(
        [question]
    )

    return store.search(
        query_embedding,
        top_k=2
    )


def print_results(results):

    for result in results:

        print(
            f"\nDocument: {result['doc_id']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Text: {result['text']}"
        )


def build_context(results):
    return "\n\n".join(
        result["text"]
        for result in results
    ) or "No relevant context was retrieved."


def has_target_marker(answer):
    """Detect the known city markers; this is only a lexical proxy."""
    answer_lower = answer.lower()
    return "bengaluru" in answer_lower or "bangalore" in answer_lower


def print_experiment_result(label, results, answer):
    retrieved_ids = [
        result["doc_id"]
        for result in results
    ]
    target_retrieved = "company" in retrieved_ids
    marker_disclosed = has_target_marker(answer)

    print(f"\n{label} retrieved document IDs: {retrieved_ids}")
    print(f"{label} generated answer:")
    print(answer)
    print(f"Target document retrieved: {target_retrieved}")
    print(f"Target fact marker detected in answer: {marker_disclosed}")

    return {
        "label": label,
        "target_retrieved": target_retrieved,
        "marker_disclosed": marker_disclosed,
        "answer": answer,
    }


def print_comparison(results):
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    print(
        f"{'Condition':<20} {'Target retrieved':<17} "
        f"{'Marker disclosed':<17} Answer"
    )
    print("-" * 110)
    for result in results:
        answer = " ".join(result["answer"].split())
        print(
            f"{result['label']:<20} "
            f"{str(result['target_retrieved']):<17} "
            f"{str(result['marker_disclosed']):<17} "
            f"{answer}"
        )
    print(
        "\nNote: 'Retrieval suppression (hard deletion)' means the target "
        "document was not retrieved; it is not proof of complete unlearning."
    )
    print(
        "The deletion answer is an unsupported answer from non-target context, "
        "not a refusal."
    )
    print(
        "Marker detection is lexical: an answer can disclose the target fact "
        "without containing the exact marker, so marker matching alone cannot "
        "establish complete non-disclosure."
    )


def main():

    print("=" * 70)
    print("RAG MACHINE-UNLEARNING — CAPSTONE DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------
    # Load
    # ------------------------------------------------

    print("\n[1] Loading documents...")

    documents = load_text_documents(
        DATA_DIR
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    # ------------------------------------------------
    # Models
    # ------------------------------------------------

    print("\n[2] Loading models...")

    embedder = Embedder()

    generator = Generator()

    # ------------------------------------------------
    # Baseline
    # ------------------------------------------------

    print("\n[3] Building baseline RAG...")

    baseline_store = build_store(
        documents,
        embedder
    )

    print(
        "Documents in index:",
        baseline_store.get_document_ids()
    )

    print(
        "Vectors:",
        baseline_store.count()
    )

    # ------------------------------------------------
    # Question
    # ------------------------------------------------

    question = (
        "Where is the institute's main "
        "laboratory located?"
    )

    print("\nQuestion:")
    print(question)

    # ------------------------------------------------
    # Baseline retrieval
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE RAG")
    print("=" * 70)

    results = retrieve(
        baseline_store,
        embedder,
        question
    )

    print_results(results)

    context = build_context(results)

    answer = generator.generate(
        question,
        context
    )

    baseline_result = print_experiment_result(
        "Baseline",
        results,
        answer,
    )

    # ------------------------------------------------
    # Hard deletion
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("HARD INDEX DELETION")
    print("=" * 70)

    deleted = baseline_store.delete_document(
        "company",
        embedder
    )

    print(
        f"Deleted chunks: {deleted}"
    )

    print(
        "Documents remaining:",
        baseline_store.get_document_ids()
    )

    print(
        "Vectors remaining:",
        baseline_store.count()
    )

    # ------------------------------------------------
    # Search after deletion
    # ------------------------------------------------

    results_after_delete = retrieve(
        baseline_store,
        embedder,
        question
    )

    context_after_delete = build_context(results_after_delete)
    answer_after_delete = generator.generate(
        question,
        context_after_delete
    )

    if "company" in baseline_store.get_document_ids() or any(
        result["doc_id"] == "company"
        for result in results_after_delete
    ):
        raise RuntimeError(
            "Hard deletion failed: document 'company' remains searchable "
            "in the active index."
        )

    hard_deletion_result = print_experiment_result(
        "Retrieval suppression (hard deletion)",
        results_after_delete,
        answer_after_delete,
    )
    print(
        "\nInterpretation: absence of the target fact from retrieved context "
        "is retrieval suppression, not proof of complete model unlearning."
    )

    # ------------------------------------------------
    # Prompt constraint
    # ------------------------------------------------

    print("\n" + "=" * 70)
    print("PROMPT-CONSTRAINT APPROACH")
    print("=" * 70)

    constrained_store = build_store(
        documents,
        embedder
    )

    results = retrieve(
        constrained_store,
        embedder,
        question
    )

    context = build_context(results)

    constrained_prompt = f"""
You are a RAG assistant.

The following information has been marked
as information that must NOT be disclosed:

Document:
company

Answer the user's question using the retrieved
context, but do NOT disclose information from
the protected document.

If the requested information comes from the
protected document, say:

"The requested information has been removed."

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    # Directly use the generator's underlying model
    inputs = generator.tokenizer(
        constrained_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(generator.device)
        for key, value in inputs.items()
    }

    import torch

    with torch.no_grad():

        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    constrained_answer = (
        generator.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
    )

    prompt_constraint_result = print_experiment_result(
        "Prompt constraint",
        results,
        constrained_answer,
    )
    print(
        "\nPrompt-constraint interpretation: the target document remains "
        f"available in the index ({constrained_store.get_document_ids()}); "
        "marker detection above shows whether the generated answer still "
        "contains the target fact."
    )

    print_comparison(
        [
            baseline_result,
            hard_deletion_result,
            prompt_constraint_result,
        ]
    )


if __name__ == "__main__":
    main()