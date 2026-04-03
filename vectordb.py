import os
import json
from typing import Optional, List, Dict, Any, Tuple
import re
import pandas as pd
import requests
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from dotenv import load_dotenv
# =========================
# Config
# =========================
DATA_PATH = "ATBe_2024_test.csv"  # put in same folder, or change path
INDEX_PATH = "faiss_index_test"

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Drop junk index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Loaded Dataset
    
    return df

def row_to_text(row: pd.Series) -> str:
    """
    Turn a row into a semantically searchable text chunk.
    We keep ALL the important fields so embeddings match natural language queries.
    """
    parts = [
        f"ATB release year: {row.get('atb_year')}",
        f"Technology: {row.get('technology')}",
        f"Display name: {row.get('display_name')}",
        f"Scenario: {row.get('scenario')}",
        f"Core metric parameter: {row.get('core_metric_parameter')}",
        f"Case: {row.get('core_metric_case')}",
        f"Tax credit case: {row.get('tax_credit_case')}",
        f"CRP years: {row.get('crpyears')}",
        f"Scale: {row.get('scale')}",
        f"Maturity: {row.get('maturity')}",
        f"Tech detail: {row.get('techdetail')} / {row.get('techdetail2')}",
        f"Resource detail: {row.get('resourcedetail')}",
        f"Metric year (core_metric_variable): {row.get('core_metric_variable')}",
        f"Units: {row.get('units')}",
        f"Value: {row.get('value')}",
        f"Metric key: {row.get('core_metric_key')}",
        f"Technology alias: {row.get('technology_alias')}",
    ]
    return "\n".join([p for p in parts if p and str(p) != "nan"])


def build_vectorstore(df: pd.DataFrame) -> FAISS:

    # Load existing index if it exists
    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index from disk...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
        return vs
    
    # Otherwise build index
    print("Building FAISS index from scratch... (this may take a while)\n")

    docs: List[Document] = []

    for idx, row in df.iterrows():
        text = row_to_text(row)

        metadata = {
            "row_index": int(idx),
            "technology": str(row.get("technology")),
            "display_name": str(row.get("display_name")),
            "scenario": str(row.get("scenario")),
            "core_metric_parameter": str(row.get("core_metric_parameter")),
            "core_metric_variable": str(row.get("core_metric_variable")),
            "atb_year": str(row.get("atb_year")),
        }
        docs.append(Document(page_content=text, metadata=metadata))
        
        # Progress check of vectorization every 10k rows
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} rows...")

    # HuggingFace embeddings (local model download)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embeddings)

    # Saving index locally
    print("Saving FAISS index to disk...")
    vs.save_local(INDEX_PATH)

    return vs


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    vs = build_vectorstore(df)

    print(vs)