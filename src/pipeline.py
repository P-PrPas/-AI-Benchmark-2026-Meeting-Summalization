import os
import json
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

from .data_loader import load_dataset, get_document_paragraphs, get_queries, get_paragraph_by_id
from .embedder import Embedder, FAISSRetriever
from .retrieval import retrieve_references, select_references_from_retrieved
from .generator import Generator
from . import config


class SummarizationPipeline:
    """End-to-end pipeline for Thai parliamentary meeting summarization."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        generator: Optional[Generator] = None,
        retrieval_top_k: int = 10,
        reference_top_n: int = 3
    ):
        self.embedder = embedder or Embedder()
        self.generator = generator or Generator()
        self.retriever = FAISSRetriever(self.embedder)
        self.retrieval_top_k = retrieval_top_k
        self.reference_top_n = reference_top_n

    def load_and_index(self, data_path: str):
        """Load documents and index them for retrieval."""
        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Indexing {len(data['docs'])} documents...")
        for doc in tqdm(data["docs"]):
            self.retriever.index_document(doc["doc_id"], doc["paragraphs"])

        print("Indexing complete!")
        return data

    def predict_single(self, query: str, doc_id: str) -> Dict:
        """Predict abstractive and refs for a single query.

        Returns:
            Dict with keys: ID, query, doc_id, abstractive, refs
        """
        retrieved = self.retriever.retrieve(doc_id, query, self.retrieval_top_k)

        top_refs = retrieve_references(
            self.retriever, doc_id, query, self.reference_top_n
        )

        abstractive = self.generator.generate(query, retrieved)

        return {
            "query": query,
            "doc_id": doc_id,
            "abstractive": abstractive,
            "refs": top_refs
        }

    def predict_batch(
        self,
        queries: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """Predict for multiple queries.

        Args:
            queries: List of query dicts with 'ID', 'doc_id', 'query' keys

        Returns:
            List of prediction dicts
        """
        predictions = []
        iterator = tqdm(queries, desc="Predicting") if show_progress else queries

        for q in iterator:
            result = self.predict_single(q["query"], q["doc_id"])
            result["ID"] = q["ID"]
            predictions.append(result)

        return predictions

    def predict_test(self) -> List[Dict]:
        """Predict on test set."""
        test_data = load_dataset("test")
        queries = test_data["queries"]
        return self.predict_batch(queries)

    def create_submission(
        self,
        predictions: List[Dict],
        output_path: str = None
    ) -> pd.DataFrame:
        """Create submission DataFrame from predictions.

        Args:
            predictions: List of prediction dicts
            output_path: Optional path to save CSV

        Returns:
            DataFrame in submission format
        """
        rows = []
        for p in predictions:
            rows.append({
                "ID": p["ID"],
                "abstractive": p["abstractive"],
                "refs": ",".join(p["refs"]) if isinstance(p["refs"], list) else str(p["refs"])
            })

        df = pd.DataFrame(rows)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Submission saved to {output_path}")

        return df

    def run_train_evaluation(self) -> Dict:
        """Run evaluation on training set."""
        train_data = load_dataset("train")
        queries = train_data["queries"]

        predictions = self.predict_batch(queries)

        return predictions


def main():
    """Main function to run the pipeline."""
    pipeline = SummarizationPipeline()

    print("=" * 60)
    print("CAMNET-P Thai Summarization Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading and indexing data...")
    train_data = pipeline.load_and_index(config.TRAIN_PATH)

    print(f"\nLoaded {len(train_data['docs'])} documents with {len(train_data['queries'])} queries")

    print("\n[2/3] Running predictions on train set...")
    predictions = pipeline.run_train_evaluation()

    print("\n[3/3] Creating submission...")
    submission = pipeline.create_submission(
        predictions,
        os.path.join(config.OUTPUT_DIR, "submission.csv")
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return submission


if __name__ == "__main__":
    main()