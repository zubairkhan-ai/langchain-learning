from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [

    Document(
        page_content="LangChain helps developers build LLM applications easily."
    ),

    Document(
        page_content="ChromaDB is a vector database optimized for LLM-based applications."
    ),

    Document(
        page_content="Embeddings convert text into high-dimensional vectors."
    ),

    Document(
        page_content="OpenAI provides powerful embedding models."
    ),

]


# ============================================================
# 2. EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. CREATE VECTOR STORE
# ============================================================

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)


# ============================================================
# 4. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# ============================================================
# 5. QUERY
# ============================================================

query = "What is Chroma used for?"


# ============================================================
# 6. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

# results = retriever.invoke(query)

results=vectorstore.similarity_search(query,k=2)
# ============================================================
# 7. PRINT RESULTS
# ============================================================

for i, doc in enumerate(results):

    print(f"\nResult --- {i + 1} ---")

    print(doc.page_content)