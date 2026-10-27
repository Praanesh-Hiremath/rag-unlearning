import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import chunk_text
from demo import DATA_DIR, build_store
from ingestion import load_text_documents
from run_benchmark import classify_target_answer, make_summary
from vector_store import VectorStore


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("bengaluru" in lowered),
                float("2018" in lowered),
                float("2024" in lowered),
                float("ranking" in lowered),
            ])
        return np.asarray(vectors, dtype="float32")


class CoreBehaviourTests(unittest.TestCase):
    def test_demo_data_path_and_sample_ingestion_are_project_root_based(self):
        documents = load_text_documents(DATA_DIR)

        self.assertEqual(DATA_DIR, Path(__file__).resolve().parents[1] / "data" / "original")
        self.assertEqual(
            {document["doc_id"] for document in documents},
            {"company", "project"},
        )

    def test_demo_build_store_rejects_empty_documents(self):
        with self.assertRaisesRegex(ValueError, "data.original"):
            build_store([], FakeEmbedder())

    def test_demo_build_store_requires_two_dimensional_embeddings_per_chunk(self):
        class BadEmbedder:
            def encode(self, texts):
                return np.asarray([1.0, 2.0], dtype="float32")

        documents = load_text_documents(DATA_DIR)

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            build_store(documents, BadEmbedder())

    def test_demo_store_deletion_removes_target_from_active_index(self):
        documents = load_text_documents(DATA_DIR)
        embedder = FakeEmbedder()
        store = build_store(documents, embedder)

        self.assertIn("company", store.get_document_ids())
        self.assertGreater(store.count(), 0)
        self.assertGreater(store.delete_document("company", embedder), 0)
        self.assertNotIn("company", store.get_document_ids())
        self.assertTrue(
            all(
                result["doc_id"] != "company"
                for result in store.search(
                    embedder.encode(["main laboratory Bengaluru"]),
                    top_k=10,
                )
            )
        )

    def test_chunking_preserves_overlap(self):
        chunks = chunk_text("one two three four five", chunk_size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five"])

    def test_hard_deletion_rebuilds_index_without_target(self):
        embedder = FakeEmbedder()
        chunks = [
            {"doc_id": "keep", "text": "document ranking", "source": "test"},
            {"doc_id": "remove", "text": "Bengaluru", "source": "test"},
        ]
        vectors = embedder.encode([chunk["text"] for chunk in chunks])
        store = VectorStore(vectors.shape[1])
        store.add(vectors, chunks)

        self.assertEqual(store.delete_document("remove", embedder), 1)
        self.assertEqual(store.get_document_ids(), ["keep"])
        self.assertEqual(store.count(), 1)
        results = store.search(embedder.encode(["Bengaluru"]), top_k=2)
        self.assertTrue(all(item["doc_id"] != "remove" for item in results))

    def test_answer_classifier_distinguishes_refusal_and_wrong_answer(self):
        self.assertEqual(
            classify_target_answer("The information is unavailable.", "2018")[0],
            "abstention",
        )
        self.assertEqual(
            classify_target_answer("The answer is 2024.", "2018")[0],
            "unsupported_or_wrong",
        )
        self.assertEqual(
            classify_target_answer("It was 2018.", "2018")[0],
            "leak",
        )

    def test_summary_uses_explicit_test_type_and_denominators(self):
        rows = [
            {
                "condition": "baseline",
                "test_type": "target",
                "leakage_detected": "1",
                "target_doc_retrieved": "1",
                "target_outcome": "leak",
            },
            {
                "condition": "baseline",
                "test_type": "control",
                "control_marker_match": "1",
                "target_doc_retrieved": "0",
                "target_outcome": "",
            },
        ]
        summary = make_summary(rows)[0]
        self.assertEqual(summary["target_queries"], 1)
        self.assertEqual(summary["target_leaks"], 1)
        self.assertEqual(summary["control_queries"], 1)
        self.assertEqual(summary["control_marker_matches"], 1)


if __name__ == "__main__":
    unittest.main()
