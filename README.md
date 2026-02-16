# cs675-agentic-construction
We are planning on developing a resource management agent that can take cost/resource inputs and produce plans for future developments, for example in power infrastructure or construction developments.

To run the draft1 file initally you will have to downgrade to python 3.13. You will also have to generate your own API key for OpenRouter and declare it in
your $env. 

We choose OpenRouter LLM for its wide variety of model options. draft1 currently uses HuggingFaceEmbedddings because it was a free option. This could be improved
once we have a basic working proof of concept. 1build API has been causing issues, so we are currently looking into other API calls/ datasets aswell for the 
construction cost. 
