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
DATA_PATH = "ATBe_2024.csv"  # put in same folder, or change path
INDEX_PATH = "faiss_index"
OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
load_dotenv()

# =========================
# Utilities: dataset -> docs
# =========================
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Drop junk index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
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
- If you cannot find anything, ask the user to rephrase with a technology name like in the dataset.
- Never call the same tool more than once for the same query unless absolutely necessary. 
- Prefer to produce a final answer after one estimate.

Tool Usage:
When you need a tool, output EXACTLY a JSON object like:
{"tool":"retrieve","args":{...}}
or
{"tool":"estimate","args":{...}}

Tool Results:
Tool results will appear as a message starting with:
TOOL_RESULT (toolname): {...}

Use that data to produce the final answer.

Otherwise output normal helpful text.
"""
def parse_xml_tool_call(text: str):
    if "<tool_call>" not in text:
        return None

    function_match = re.search(r"<function=(.*?)>", text)
    if not function_match:
        return None

    tool_name = function_match.group(1).strip()

    params = {}
    param_matches = re.findall(
        r"<parameter=(.*?)>\s*(.*?)\s*</parameter>",
        text,
        re.DOTALL
    )

    for name, value in param_matches:
        params[name.strip()] = value.strip()

    return {"tool": tool_name, "args": params}


def run_agent(df: pd.DataFrame, vs: FAISS) -> None:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Cost Estimation Agent (OpenRouter + stepfun/step-3.5-flash:free)")
    print("Type 'quit' to exit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break

        messages.append({"role": "user", "content": user})

        tool_calls_made = 0

        for _ in range(6):  # max tool iterations
            assistant = openrouter_chat(messages).strip()

            # Try JSON tool call
            try:
                tool_call = json.loads(assistant)
            except:
                tool_call = None

            # Try XML tool call
            if tool_call is None:
                print("XML tool call")
                tool_call = parse_xml_tool_call(assistant)
                print(tool_call)

            # If a tool was requested
            if isinstance(tool_call, dict) and "tool" in tool_call:
                tool = tool_call["tool"]
                args = tool_call["args"]

                # increment tool call count
                tool_calls_made += 1

                if tool_calls_made >= 5:
                    messages.append({
                        "role": "user",
                        "content": "Provide a final answer based on the retrieved data. Do not call any more tools."
                    })
                    continue

                if tool == "retrieve":
                    # ensure k is an int (LLM returning string instead of int)
                    if "k" in args:
                        try:
                            args["k"] = int(args["k"])
                        except:
                            args["k"] = 8  # fallback default

                    req = RetrieveRequest(**args)
                    result = tool_retrieve(vs, req)

                elif tool == "estimate":
                    # ensure k is an int (LLM returning string instead of int)
                    if "year" in args and args["year"] is not None:
                        try:
                            args["year"] = int(args["year"])
                        except:
                            args["year"] = None
                    req = EstimateRequest(**args)
                    result = tool_estimate(df, req)

                else:
                    result = {"error": f"Unknown tool: {tool}"}

                messages.append({"role": "assistant", "content": assistant})
                messages = messages[-12:]
                messages.append({
                "role": "user",
                "content": f"TOOL_RESULT ({tool}): {json.dumps(result)}"
            })
                messages = messages[-12:]

                continue  # loop again so LLM can interpret tool output

            # Normal response
            messages.append({"role": "assistant", "content": assistant})
            messages = messages[-12:]
            print(f"\nAgent: {assistant}\n")
            break


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    print("Loaded dataset into DataFrame")
    vs = build_vectorstore(df)
    run_agent(df, vs)