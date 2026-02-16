import os
from typing import List

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()



OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ONEBUILD_API_KEY = os.getenv("ONEBUILD_API_KEY")
if ONEBUILD_API_KEY:
    ONEBUILD_API_KEY = ONEBUILD_API_KEY.strip()  # removes hidden whitespace/newlines

print("ONEBUILD_API_KEY repr:", repr(ONEBUILD_API_KEY))
print("ONEBUILD_API_KEY length:", len(ONEBUILD_API_KEY) if ONEBUILD_API_KEY else None)



if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in your shell or .env")
if not ONEBUILD_API_KEY:
    raise RuntimeError("Missing ONEBUILD_API_KEY. Set it in your shell or .env")

ONEBUILD_BASE_URL = "https://gateway-external.1build.com"

def onebuild_search(
    search_term: str,
    state: str = "California",
    county: str = "Los Angeles County",
    limit: int = 5,
):
    # NOTE: gateway expects POST to the base URL
    url = ONEBUILD_BASE_URL.rstrip("/") + "/"
    headers = {
        "1build-api-key": ONEBUILD_API_KEY,
        "content-type": "application/json",
    }

    gql = """
    query Sources($input: SourceSearchInput!) {
      sources(input: $input) {
        nodes {
          id
          name
          calculatedUnitRateUsdCents
          laborRateUsdCents
          materialRateUsdCents
        }
      }
    }
    """

    variables = {
        "input": {
            "state": state,
            "county": county,
            "searchTerm": search_term,
            "page": {"limit": limit},
        }
    }

    r = requests.post(url, headers=headers, json={"query": gql, "variables": variables}, timeout=30)

    # Always print a small debug slice for now (you can comment out later)
    print("1build status:", r.status_code)
    print("1build body head:", r.text[:300])

    r.raise_for_status()
    payload = r.json()

    # GraphQL can return 200 with errors
    if "errors" in payload and payload["errors"]:
        # ---- Mock fallback so your agent still works for CS 675 demos ----
        # If the service says UNAUTHENTICATED/invalid key, we return a mock unit-rate.
        # This keeps your tool + RAG plumbing functional and lets you explain the limitation.
        msg = str(payload["errors"])
        if "UNAUTHENTICATED" in msg or "invalid API key" in msg:
            return [{
                "id": "mock-1",
                "name": f"Mock unit-rate: {search_term} (CA / {county})",
                "calculatedUnitRateUsdCents": 250000,  # $2,500.00
                "laborRateUsdCents": 120000,           # $1,200.00
                "materialRateUsdCents": 130000,        # $1,300.00
            }]
        # Otherwise, surface the real GraphQL errors
        raise RuntimeError(payload["errors"])

    data = payload.get("data") or {}
    sources = data.get("sources") or {}
    return sources.get("nodes") or []



# LLM via OpenRouter
llm = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Chroma vector store
vectorstore = Chroma(
    collection_name="demo_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def seed_if_empty() -> None:
   docs: List[Document] = []



seed_if_empty()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer using ONLY the provided context. "
         "The context may include RAG snippets and 1build tool JSON. "
         "If the context doesn't contain enough cost info, say what is missing."
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)



def ask(question: str, k: int = 4) -> str:
    # 1) Retrieve from Chroma (memory / RAG)
    results = vectorstore.similarity_search(question, k=k)
    rag_context = "\n\n".join(
        f"- {d.page_content}" for d in results
    ) if results else ""

    # 2) Always try 1build (tool call)
    tool_context = ""
    try:
        nodes = onebuild_search(question)  # GraphQL call
        if nodes:
            tool_context = (
                "\n\n1build matches:\n"
                + "\n".join(
                    f"- {n.get('name', 'Unknown')}: "
                    f"${(n.get('calculatedUnitRateUsdCents', 0) / 100):.2f} "
                    f"(labor ${(n.get('laborRateUsdCents', 0) / 100):.2f}, "
                    f"materials ${(n.get('materialRateUsdCents', 0) / 100):.2f})"
                    for n in nodes
                )
            )
        else:
            tool_context = "\n\n(1build returned no matches)"
    except Exception as e:
        tool_context = f"\n\n(1build lookup failed: {e})"

    # 3) Build final context
    context = (rag_context + tool_context).strip() or "(no matches)"

    # 4) Ask the LLM
    resp = (prompt | llm).invoke({
        "question": question,
        "context": context
    })

    return resp.content


if __name__ == "__main__":
    print("Type a question (or 'exit' to quit).")
    while True:
        q = input("\n> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("\n" + ask(q))
        
    






    # Optional: quick 1build test
    # print(onebuild_search("concrete"))
