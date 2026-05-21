# cs675-agentic-construction
This agent is designed to query through the NREL Annual Technology baseline (ATB) 2024 dataset (~570k rows) using natural language to provide data-backed responses from the dataset. 

Dataset was acquired here: https://data.openei.org/submissions/6006

It currently has three tools implemented: Retrieve, Estimate, and Forecast.

Retrieve: Semantically search over the FAISS index –used when the query is vague or agent needs to explore available metrics.
Estimate: Filters the DataFrame by technology, metric, scenario, and year, then returns summary statistics for a point-in-time lookup
Forecast: Fits a linear trend across all historical metric-years for a given slice and projects to target year, reporting R squared and extrapolation distance.

You will have to generate your own API key for OpenRouter and declare it in your $env. 

We choose OpenRouter LLM for its wide variety of model options. The most updated program is cost_agent_3tool.

As a reminder, to use a virtual environment to run this project (on Windows):
1. Check that Python 3 is installed:
    python --version
        You should see something like Python 3.x.x
        If not, install from python.org
2. Create a Virtual Environment
    Run this command in the project root directory:
    python -m venv .venv
3. Activate the virtual environment
    run this command in project root directory:
    .venv\Scripts\Activate.ps1
        You should see (.venv) in the beginning of your terminal prompt
4. Upgrade pip (Optional but recommended)
    Run the command:
    pip install --upgrade pip
5. Install project dependencies
    Run the command:
    pip install -r requirements.txt
    This will install all the dependencies for this project.
