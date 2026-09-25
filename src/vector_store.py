import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.chunks = []

    def add(self, embeddings, chunks):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.index.ntotal == 0:
            return []

        top_k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            result = self.chunks[index].copy()

            result["score"] = float(score)

            results.append(result)

        return results

    def delete_document(
        self,
        doc_id,
        embedder
    ):
        """
        Remove a document from the active index.
        """

        remaining_chunks = [
            chunk
            for chunk in self.chunks
            if chunk["doc_id"] != doc_id
        ]

        deleted_chunks = (
            len(self.chunks)
            - len(remaining_chunks)
        )

        if deleted_chunks == 0:
            return 0

        # Rebuild index
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.chunks = []

        if remaining_chunks:

            texts = [
                chunk["text"]
                for chunk in remaining_chunks
            ]

            embeddings = embedder.encode(
                texts
            )

            self.add(
                embeddings,
                remaining_chunks
            )

        return deleted_chunks

    def get_document_ids(self):

        return sorted(
            set(
                chunk["doc_id"]
                for chunk in self.chunks
            )
        )

    def count(self):

        return self.index.ntotal