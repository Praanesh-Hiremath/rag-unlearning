from pathlib import Path


def load_text_documents(directory: str):
    """
    Load all .txt files from a directory.

    Returns:
        List of dictionaries containing document text and metadata.
    """
    documents = []

    directory_path = Path(directory)

    for file_path in directory_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "doc_id": file_path.stem,
            "source": str(file_path),
            "text": text,
        })

    return documents


if __name__ == "__main__":
    documents = load_text_documents("data/original")

    for document in documents:
        print("=" * 60)
        print("Document:", document["doc_id"])
        print(document["text"])