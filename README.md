# LangChain Learning and Practice

This repository contains my practice code while learning **LangChain** and building LLM-based applications using Python.

The examples cover prompt engineering, language models, document processing, embeddings, vector stores, retrieval techniques, tools, and agents.

## Topics Covered

- Prompt templates
- Output parsers
- Runnable sequences
- Runnable parallel
- Chat models
- Hugging Face models
- TinyLlama
- Text embeddings
- Document loaders
- Text splitters
- FAISS vector store
- ChromaDB
- Similarity search
- Maximum Marginal Relevance
- MultiQueryRetriever
- Custom tools
- DuckDuckGo search
- ReAct-style agents

## Technologies Used

- Python
- LangChain
- Hugging Face Transformers
- TinyLlama
- PyTorch
- FAISS
- ChromaDB
- Sentence Transformers
- DuckDuckGo Search

## Project Structure

```text
langchain-learning/
│
├── chains/
├── ChatModels/
├── document_loader/
├── EmbeddedModels/
├── langchain_prompt/
├── langchain_runnables/
├── langchain_tools/
├── langchain_tool_calling/
├── output_parser/
├── retrieval/
├── runnables/
├── text_splitters/
├── vector_stores/
├── with_structure_output/
├── agents_in_langchain/
├── LLMS/
│
├── README.md
├── requirements.txt
└── .gitignore


Installation
1. Clone the Repository
git clone https://github.com/zubairkhan-ai/langchain-learning.git
2. Open the Project Folder
cd langchain-learning
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

On Windows PowerShell:

venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
Running the Examples

Run any Python file from its relevant folder.

For example:

python retrieval/mmr_retrieval.py

Another example:

python langchain_tool_calling/react_agent.py

The exact filename may differ depending on the example.

Models and Embeddings

This repository mainly uses local Hugging Face models, including TinyLlama.

For text embeddings, the examples use Sentence Transformers models such as:

sentence-transformers/all-MiniLM-L6-v2

Some examples may download models from Hugging Face during the first run.

Important Notes

This repository contains educational practice code created while learning LangChain.

Some examples require an internet connection.

Some tools may require external API keys.

API keys and secret credentials are not included in this repository.

Virtual environments and downloaded model files are excluded from GitHub.

Local models may take additional time to download during the first run.

Learning Objective

The purpose of this repository is to understand how LangChain can be used to build applications involving language models, retrieval systems, vector databases, tools, and agents.



