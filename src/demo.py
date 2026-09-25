from ingestion import load_text_documents
from chunking import chunk_documents
from embeddings import Embedder
from vector_store import VectorStore
from generator import Generator


DATA_DIR = "../data/original"

def build_store(documents, embedder):

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        overlap=20
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.encode(texts)

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


def main():

    print("=" * 70)
    print("RAG UNLEARNING — PHASE 2 DEMONSTRATION")
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

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    answer = generator.generate(
        question,
        context
    )

    print("\nAnswer:")
    print(answer)

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

    print("\nRetrieval AFTER deletion:")

    print_results(
        results_after_delete
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

    context = "\n\n".join(
        result["text"]
        for result in results
    )

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

    print(
        "\nVector still exists:"
    )

    print(
        constrained_store.get_document_ids()
    )

    print(
        "\nPrompt-constrained answer:"
    )

    print(
        constrained_answer
    )


if __name__ == "__main__":
    main()