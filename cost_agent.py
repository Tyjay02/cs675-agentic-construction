import os
import json
from typing import Optional, List, Dict, Any, Tuple

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
OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
load_dotenv()


# =========================
# Utilities: dataset -> docs
# =========================
def load_dataset(path: str) -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_csv(path)
    # Drop junk index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    print("Loaded dataset.")
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
    print("Building vectorstore...")
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

    # HuggingFace embeddings (local model download)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embeddings)
    print("Done building vectorstore.")
    return vs


# =========================
# Tools the agent can use
# =========================
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural language query about technology/cost/performance.")
    k: int = Field(8, description="How many similar rows to retrieve.")


class EstimateRequest(BaseModel):
    technology_contains: str = Field(..., description="Substring to match in technology or display_name (e.g., 'Offshore', 'Hydropower', 'PV').")
    metric: str = Field(..., description="core_metric_parameter to estimate (e.g., 'Fixed O&M', 'CAPEX', 'CFC').")
    scenario: Optional[str] = Field(None, description="Scenario filter (Conservative/Moderate/Advanced).")
    year: Optional[int] = Field(None, description="core_metric_variable year filter, like 2026.")
    core_metric_case: Optional[str] = Field(None, description="Filter like Market or R&D.")
    tax_credit_case: Optional[str] = Field(None, description="Filter like ITC/PTC/None.")


def tool_retrieve(vs: FAISS, req: RetrieveRequest) -> Dict[str, Any]:
    results = vs.similarity_search(req.query, k=req.k)
    out = []
    for d in results:
        out.append(
            {
                "metadata": d.metadata,
                "text": d.page_content,
            }
        )
    return {"matches": out}


def tool_estimate(df: pd.DataFrame, req: EstimateRequest) -> Dict[str, Any]:
    # Normalize
    tech_q = req.technology_contains.lower().strip()
    metric_q = req.metric.lower().strip()

    # Candidate rows
    mask = (
        df["technology"].astype(str).str.lower().str.contains(tech_q, na=False)
        | df["display_name"].astype(str).str.lower().str.contains(tech_q, na=False)
    )
    sub = df[mask].copy()

    if sub.empty:
        return {"error": f"No rows matched technology_contains='{req.technology_contains}'."}

    # Metric filter (loose contains so 'Fixed O&M' matches)
    sub = sub[sub["core_metric_parameter"].astype(str).str.lower().str.contains(metric_q, na=False)]
    if sub.empty:
        return {"error": f"No rows matched metric='{req.metric}' for technology_contains='{req.technology_contains}'."}

    # Optional filters
    if req.scenario:
        sub = sub[sub["scenario"].astype(str).str.lower() == req.scenario.lower()]
    if req.year is not None:
        sub = sub[pd.to_numeric(sub["core_metric_variable"], errors="coerce") == req.year]
    if req.core_metric_case:
        sub = sub[sub["core_metric_case"].astype(str).str.lower() == req.core_metric_case.lower()]
    if req.tax_credit_case:
        # handle NaN as "None"
        tcc = req.tax_credit_case.lower()
        if tcc in ["none", "no", "nan"]:
            sub = sub[sub["tax_credit_case"].isna()]
        else:
            sub = sub[sub["tax_credit_case"].astype(str).str.lower() == tcc]

    if sub.empty:
        return {"error": "Filters removed all rows. Try relaxing scenario/year/case filters."}

    # Convert values to numeric
    sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value_num"])
    if sub.empty:
        return {"error": "Matched rows had non-numeric values only."}

    # Simple estimate: mean + min/max, return exemplars
    estimate = {
        "count": int(len(sub)),
        "mean": float(sub["value_num"].mean()),
        "min": float(sub["value_num"].min()),
        "max": float(sub["value_num"].max()),
        "units": sub["units"].dropna().unique().tolist(),
        "examples": [],
    }

    # Provide up to 5 example rows
    for _, r in sub.head(5).iterrows():
        estimate["examples"].append(
            {
                "display_name": r.get("display_name"),
                "technology": r.get("technology"),
                "scenario": r.get("scenario"),
                "metric": r.get("core_metric_parameter"),
                "metric_year": r.get("core_metric_variable"),
                "core_metric_case": r.get("core_metric_case"),
                "tax_credit_case": r.get("tax_credit_case"),
                "value": float(r.get("value_num")),
                "units": r.get("units"),
            }
        )

    return estimate


# =========================
# OpenRouter chat client
# =========================
def openrouter_chat(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter:
        "HTTP-Referer": "http://localhost",
        "X-Title": "ATB Cost Estimation Agent",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# =========================
# A minimal "tool-using" loop (lightweight agent)
# =========================
SYSTEM_PROMPT = """You are a cost estimation assistant using an ATB-style dataset.
You have two tools:
1) retrieve(query, k): retrieves the most relevant rows as text with metadata
2) estimate(technology_contains, metric, scenario?, year?, core_metric_case?, tax_credit_case?): returns summary stats from filtered rows

Rules:
- If the user asks “what is the cost/performance of X”, use retrieve first if unclear, then estimate.
- Always cite which technology, scenario, year, and metric you used.
- If units are missing, say units are not provided in the sheet and treat values as dataset-native units for that metric.
- If you cannot find anything, ask the user to rephrase with a technology name like in the dataset (e.g., Offshore Wind, Hydropower, Utility-Scale PV-Plus-Battery).
When you need a tool, output EXACTLY a JSON object with:
{"tool":"retrieve","args":{...}}
or
{"tool":"estimate","args":{...}}
Otherwise output normal helpful text.
"""


def run_agent(df: pd.DataFrame, vs: FAISS) -> None:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Cost Estimation Agent (OpenRouter + stepfun/step-3.5-flash:free)")
    print("Type 'quit' to exit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break

        messages.append({"role": "user", "content": user})

        for _ in range(6):  # tool loop cap
            assistant = openrouter_chat(messages).strip()
            # Try interpret tool call JSON
            tool_call = None
            try:
                tool_call = json.loads(assistant)
            except json.JSONDecodeError:
                tool_call = None

            if isinstance(tool_call, dict) and "tool" in tool_call and "args" in tool_call:
                tool = tool_call["tool"]
                args = tool_call["args"]

                if tool == "retrieve":
                    req = RetrieveRequest(**args)
                    result = tool_retrieve(vs, req)
                elif tool == "estimate":
                    req = EstimateRequest(**args)
                    result = tool_estimate(df, req)
                else:
                    result = {"error": f"Unknown tool: {tool}"}

                messages.append({"role": "assistant", "content": assistant})
                messages.append({"role": "tool", "content": json.dumps(result)})
                continue

            # Normal response
            messages.append({"role": "assistant", "content": assistant})
            print(f"\nAgent: {assistant}\n")
            break


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    vs = build_vectorstore(df)
    run_agent(df, vs)