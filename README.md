# cs675-agentic-construction
We are planning on developing a resource management agent that can take cost/resource inputs and produce plans for future developments, for example in power infrastructure or construction developments.

To run the draft1 file initally you will have to downgrade to python 3.13. You will also have to generate your own API key for OpenRouter and declare it in
your $env. 

We choose OpenRouter LLM for its wide variety of model options. draft1 currently uses HuggingFaceEmbedddings because it was a free option. This could be improved
once we have a basic working proof of concept. 1build API has been causing issues, so we are currently looking into other API calls/ datasets aswell for the 
construction cost. 

To use a virtual environment to run this project (on Windows):
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