from pathlib import Path

from ingestion import load_text_documents
from chunking import chunk_documents
from embeddings import Embedder
from vector_store import VectorStore
from generator import Generator


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "original"


def main():

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    print("\n[1] Loading documents...")

    documents = load_text_documents(DATA_DIR)

    print(f"Loaded {len(documents)} documents.")

    # --------------------------------------------------
    # 2. Chunk documents
    # --------------------------------------------------

    print("\n[2] Chunking documents...")

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        overlap=20,
    )

    print(f"Created {len(chunks)} chunks.")

    # --------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------

    print("\n[3] Creating embeddings...")

    embedder = Embedder()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.encode(texts)

    print("Embedding shape:", embeddings.shape)

    # --------------------------------------------------
    # 4. Build vector store
    # --------------------------------------------------

    print("\n[4] Building FAISS index...")

    dimension = embeddings.shape[1]

    vector_store = VectorStore(dimension)

    vector_store.add(
        embeddings,
        chunks,
    )

    print(
        f"Vector store contains "
        f"{vector_store.count()} vectors."
    )

    # --------------------------------------------------
    # 5. Load generator
    # --------------------------------------------------

    print("\n[5] Loading generator...")

    generator = Generator()

    # --------------------------------------------------
    # 6. Ask a question
    # --------------------------------------------------

    question = input("\nAsk a question: ")

    # --------------------------------------------------
    # 7. Embed query
    # --------------------------------------------------

    query_embedding = embedder.encode([question])

    # --------------------------------------------------
    # 8. Retrieve
    # --------------------------------------------------

    results = vector_store.search(
        query_embedding,
        top_k=3,
    )

    print("\nRetrieved chunks:")
    print("-" * 60)

    for result in results:
        print(
            f"Document: {result['doc_id']}"
        )
        print(
            f"Score: {result['score']:.4f}"
        )
        print(
            f"Text: {result['text']}"
        )
        print("-" * 60)

    # --------------------------------------------------
    # 9. Build context
    # --------------------------------------------------

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    # --------------------------------------------------
    # 10. Generate answer
    # --------------------------------------------------

    print("\nRetrieved evidence:")
    print("=" * 60)

    for result in results:
        print(f"Source: {result['doc_id']}")
        print(result["text"])
        print("=" * 60)

    answer = generator.generate(
        question,
        context,
    )

    print("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()