import os
import json
from typing import Optional, List, Dict, Any, Tuple
import re
import pandas as pd
import numpy as np
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
OPENROUTER_MODEL = "stepfun/step-3.5-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
load_dotenv()

# =========================
# ATB units lookup (units column is blank in the CSV export;
# these are the canonical units from the ATB definitions page,
# all monetary values in 2022 USD)
# =========================
UNITS_MAP = {
    "CAPEX":           "$/kW (2022 USD)",
    "OCC":             "$/kW (2022 USD)",
    "GCC":             "$/kW (2022 USD)",
    "CFC":             "$/kW (2022 USD)",
    "Fixed O&M":       "$/kW-year (2022 USD)",
    "Variable O&M":    "$/MWh (2022 USD)",
    "Capacity Factor": "fraction (0-1)",
    "LCOE":            "$/MWh (2022 USD)",
    "FCR":             "fraction (0-1)",
    "WACC":            "fraction (0-1)",
    "CRF":             "fraction (0-1)",
}

def _resolve_units(series: pd.Series, metric: str) -> list:
    """
    Return units from the dataframe column when present; fall back to
    UNITS_MAP keyed on metric name (case-insensitive substring match);
    final fallback is a pointer to the ATB definitions page.
    """
    from_data = series.dropna().unique().tolist()
    if from_data:
        return from_data
 
    metric_lower = metric.lower().strip()
    for key, unit in UNITS_MAP.items():
        if key.lower() in metric_lower or metric_lower in key.lower():
            return [unit]
 
    return ["see https://atb.nrel.gov/electricity/2024/definitions"]

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

class ForecastRequest(BaseModel):
    technology_contains: str = Field(..., description="Substring to match in technology or display_name.")
    metric: str = Field(..., description="core_metric_parameter to forecast (e.g., 'CAPEX', 'Fixed O&M').")
    target_year: int = Field(..., description="Future year to project the metric value to (e.g., 2035, 2050).")
    scenario: Optional[str] = Field(None, description="Scenario filter (Conservative/Moderate/Advanced).")
    core_metric_case: Optional[str] = Field(None, description="Filter like Market or R&D.")
    tax_credit_case: Optional[str] = Field(None, description="Filter like ITC/PTC/None.")

    
def tool_retrieve(vs: FAISS, req: RetrieveRequest) -> Dict[str, Any]:
    results = vs.similarity_search(req.query, k=req.k)
    return {
        "matches": [{"metadata": d.metadata, "text": d.page_content} for d in results]
    }

def _filter_base(df: pd.DataFrame, technology_contains: str, metric: str,
                 scenario: Optional[str], core_metric_case: Optional[str],
                 tax_credit_case: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    """Shared filtering logic for estimate and forecast."""
    tech_q = technology_contains.lower().strip()
    metric_q = metric.lower().strip()
 
    mask = (
        df["technology"].astype(str).str.lower().str.contains(tech_q, na=False)
        | df["display_name"].astype(str).str.lower().str.contains(tech_q, na=False)
    )
    sub = df[mask].copy()
    if sub.empty:
        return sub, f"No rows matched technology_contains='{technology_contains}'."
 
    sub = sub[sub["core_metric_parameter"].astype(str).str.lower().str.contains(metric_q, na=False)]
    if sub.empty:
        return sub, f"No rows matched metric='{metric}' for technology_contains='{technology_contains}'."
 
    if scenario:
        sub = sub[sub["scenario"].astype(str).str.lower() == scenario.lower()]
    if core_metric_case:
        sub = sub[sub["core_metric_case"].astype(str).str.lower() == core_metric_case.lower()]
    if tax_credit_case:
        tcc = tax_credit_case.lower()
        if tcc in ["none", "no", "nan"]:
            sub = sub[sub["tax_credit_case"].isna()]
        else:
            sub = sub[sub["tax_credit_case"].astype(str).str.lower() == tcc]
 
    if sub.empty:
        return sub, "Filters removed all rows. Try relaxing scenario/case filters."
 
    return sub, None

def tool_estimate(df: pd.DataFrame, req: EstimateRequest) -> Dict[str, Any]:
    sub, err = _filter_base(df, req.technology_contains, req.metric,
                             req.scenario, req.core_metric_case, req.tax_credit_case)
    if err:
        return {"error": err}
 
    # Year filter (point-in-time lookup)
    if req.year is not None:
        sub = sub[pd.to_numeric(sub["core_metric_variable"], errors="coerce") == req.year]
    if sub.empty:
        return {"error": "Filters removed all rows. Try relaxing scenario/year/case filters."}
 
    sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value_num"])
    if sub.empty:
        return {"error": "Matched rows had non-numeric values only."}
 
    estimate = {
        "count": int(len(sub)),
        "mean": float(sub["value_num"].mean()),
        "min": float(sub["value_num"].min()),
        "max": float(sub["value_num"].max()),
        "units": _resolve_units(sub["units"], req.metric),
        "examples": [],
    }
    resolved_units = estimate["units"][0] if estimate["units"] else "unknown"
    for _, r in sub.head(5).iterrows():
        row_units = r.get("units")
        estimate["examples"].append({
            "display_name": r.get("display_name"),
            "technology": r.get("technology"),
            "scenario": r.get("scenario"),
            "metric": r.get("core_metric_parameter"),
            "metric_year": r.get("core_metric_variable"),
            "core_metric_case": r.get("core_metric_case"),
            "tax_credit_case": r.get("tax_credit_case"),
            "value": float(r.get("value_num")),
            "units": row_units if pd.notna(row_units) else resolved_units,
        })
    return estimate

def tool_forecast(df: pd.DataFrame, req: ForecastRequest) -> Dict[str, Any]:
    """
    Fits a linear trend across all available metric-years for the matched
    technology/metric/scenario slice, then projects to target_year.
 
    Returns the projected value, the fitted slope (change per year), R²,
    the years used for fitting, and a small table of fitted vs actual values
    so the LLM can sanity-check extrapolation distance.
    """
    sub, err = _filter_base(df, req.technology_contains, req.metric,
                             req.scenario, req.core_metric_case, req.tax_credit_case)
    if err:
        return {"error": err}
 
    sub["year_num"] = pd.to_numeric(sub["core_metric_variable"], errors="coerce")
    sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["year_num", "value_num"])
 
    if sub.empty:
        return {"error": "No numeric year/value pairs found after filtering."}
    if len(sub) < 2:
        return {"error": "Need at least 2 data points to fit a trend. Try relaxing filters."}
 
    x = sub["year_num"].values
    y = sub["value_num"].values
 
    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    predicted = np.polyval(coeffs, x)
 
    # R² — how well the line describes the historical data
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = round(1 - ss_res / ss_tot, 4) if ss_tot != 0 else None
 
    projection = float(np.polyval(coeffs, req.target_year))
 
    # Last known year for extrapolation-distance context
    last_data_year = int(x.max())
    extrapolation_gap = req.target_year - last_data_year
 
    # Sample of fitted vs actual across the data range
    sample_years = sorted(set(x.tolist()))
    fit_table = [
        {"year": int(yr), "actual_mean": round(float(y[x == yr].mean()), 4),
         "fitted": round(float(np.polyval(coeffs, yr)), 4)}
        for yr in sample_years[:10]          # cap at 10 rows for brevity
    ]

    return {
        "target_year": req.target_year,
        "projected_value": round(projection, 4),
        "slope_per_year": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": r_squared,
        "data_points_used": int(len(sub)),
        "years_range": [int(x.min()), last_data_year],
        "extrapolation_gap_years": extrapolation_gap,
        "units": _resolve_units(sub["units"], req.metric),
        "fit_table": fit_table,
        "note": (
            f"Linear extrapolation {extrapolation_gap} year(s) beyond last data point ({last_data_year}). "
            "Treat projections far outside the data range with caution."
            if extrapolation_gap > 5 else
            "Target year is within or close to the data range — projection is an interpolation/short extrapolation."
        ),
    }

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
 
You have THREE tools. Choose the right one based on what the user is asking:
 
1) retrieve(query, k)
   - Use when: the user's question is vague, you need to explore what technologies or metrics exist,
     or you want to confirm field names before filtering.
 
2) estimate(technology_contains, metric, scenario?, year?, core_metric_case?, tax_credit_case?)
   - Use when: the user asks about a KNOWN or CURRENT year (e.g. "what is the 2026 CAPEX?",
     "what does X cost today?", "give me the current Fixed O&M").
   - Returns summary statistics (mean/min/max) across matched rows for that point in time.
 
3) forecast(technology_contains, metric, target_year, scenario?, core_metric_case?, tax_credit_case?)
   - Use when: the user asks about a FUTURE year, asks for a PROJECTION or TREND
     (e.g. "what will CAPEX be in 2040?", "project costs to 2035", "how will costs trend?",
     "what is the expected cost by 2050?").
   - Fits a linear trend over all available historical metric-years and projects to target_year.
   - The result includes R² and an extrapolation warning — always mention those to the user.
 
Decision rules:
- "in 2030" / "by 2040" / "predict" / "project" / "trend" / "future" → forecast
- "current" / "today" / "in 2024" / "what is" (no future year) → estimate
- Unclear technology or metric → retrieve first, then estimate or forecast
- Never call estimate AND forecast for the same query; pick one.
- Always cite technology, scenario, metric, year/target_year, and units in your final answer.
- If units are missing from results, state that units are not available in the dataset.
- If R² < 0.5 on a forecast result, warn the user the trend fit is weak.
 
Tool call format — output EXACTLY a JSON object (nothing else):
{"tool":"retrieve","args":{...}}
{"tool":"estimate","args":{...}}
{"tool":"forecast","args":{...}}
 
Tool results appear as:
TOOL_RESULT (toolname): {...}
 
Use that data to give a clear final answer. After one tool result, prefer to answer directly.
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

# =========================
# Coerce args helpers
# =========================
def _coerce_int(args: dict, key: str, fallback=None):
    if key in args and args[key] is not None:
        try:
            args[key] = int(args[key])
        except (ValueError, TypeError):
            args[key] = fallback


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

        for iteration in range(6):  # max tool iterations
            
            assistant = openrouter_chat(messages).strip()

            # 1) Strip markdown fences the model sometimes wraps JSON in
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", assistant.strip(), flags=re.DOTALL).strip()
 
            tool_call = None
            try:
                tool_call = json.loads(cleaned)
            except Exception:
                pass
 
            # 2) XML fallback (some models use <tool_call> syntax)
            if tool_call is None:
                tool_call = parse_xml_tool_call(assistant)
 
            #  Log every LLM turn clearly 
            if isinstance(tool_call, dict) and "tool" in tool_call:
                print(f"\n[Tool call] {json.dumps(tool_call, indent=2)}\n")
            else:
                # Show raw model text on non-final intermediate turns
                if iteration > 0:

                    print(f"[Model raw]: {assistant[:200]}{'...' if len(assistant) > 200 else ''}\n")

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
                    _coerce_int(args, "k", fallback=8)
                    result = tool_retrieve(vs, RetrieveRequest(**args))
 
                elif tool == "estimate":
                    _coerce_int(args, "year", fallback=None)
                    result = tool_estimate(df, EstimateRequest(**args))
 
                elif tool == "forecast":
                    _coerce_int(args, "target_year")
                    result = tool_forecast(df, ForecastRequest(**args))
 
                else:
                    result = {"error": f"Unknown tool: '{tool}'. Valid tools: retrieve, estimate, forecast."}

                result_json = json.dumps(result)
                truncated = result_json[:50] + ("..." if len(result_json) > 50 else "")
                print(f"[Tool result] ({tool}): {truncated}\n")
 
                messages.append({"role": "assistant", "content": assistant})
                messages = messages[-12:]
                messages.append({
                    "role": "user",
                    "content": f"TOOL_RESULT ({tool}): {json.dumps(result)}"
                })
                messages = messages[-12:]
                continue  # let LLM interpret the result

            # Normal response (non-tool)
            messages.append({"role": "assistant", "content": assistant})
            messages = messages[-12:]
            print(f"\nAgent: {assistant}\n")
            break


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    print("Loaded dataset into DataFrame")
    vs = build_vectorstore(df)
    run_agent(df, vs)