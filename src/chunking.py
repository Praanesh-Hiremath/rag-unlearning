def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def chunk_documents(documents, chunk_size=500, overlap=50):
    """
    Chunk every document while preserving document metadata.
    """

    chunks = []

    for document in documents:
        document_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(document_chunks):
            chunks.append({
                "chunk_id": f'{document["doc_id"]}_chunk_{index}',
                "doc_id": document["doc_id"],
                "source": document["source"],
                "text": chunk,
            })

    return chunks