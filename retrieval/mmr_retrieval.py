from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ============================================================
# Sample documents
# ============================================================

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM-based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more.")
]


# ============================================================
# Embedding model
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Create FAISS vector store
# ============================================================

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)


# ============================================================
# Create retriever using MMR
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5
    }
)


# ============================================================
# Query
# ============================================================

query = "what is langchain"


# ============================================================
# Retrieve results
# ============================================================

results = retriever.invoke(query)


# ============================================================
# Print results
# ============================================================

for i, doc in enumerate(results):

    print(f"\n----- Result {i + 1} -----")

    print(doc.page_content)