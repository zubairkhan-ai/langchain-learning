# ============================================================
# MULTI-QUERY RETRIEVER
# FAISS + HuggingFace Embeddings + TinyLlama
# ============================================================

from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.documents import Document

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from transformers import pipeline

import torch


# ============================================================
# 1. DOCUMENTS
# ============================================================

all_docs = [

    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"}
    ),

    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"}
    ),

    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"}
    ),

    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"}
    ),

    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"}
    ),

    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"}
    ),

    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"}
    ),

    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"}
    ),

    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"}
    ),

    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"}
    )

]


# ============================================================
# 2. CHECK PYTORCH
# ============================================================

print("\n============================================================")
print("ENVIRONMENT CHECK")
print("============================================================")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


# ============================================================
# 3. LOAD EMBEDDINGS
# ============================================================

print("\n============================================================")
print("[1/6] LOADING EMBEDDING MODEL")
print("============================================================")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 4. CREATE FAISS
# ============================================================

print("\n============================================================")
print("[2/6] CREATING FAISS VECTOR STORE")
print("============================================================")

vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)

print("FAISS vector store created successfully.")


# ============================================================
# 5. CREATE NORMAL RETRIEVER
# ============================================================

print("\n============================================================")
print("[3/6] CREATING SIMILARITY RETRIEVER")
print("============================================================")

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

print("Similarity retriever created successfully.")


# ============================================================
# 6. LOAD TINYLLAMA
# ============================================================

print("\n============================================================")
print("[4/6] LOADING TINYLLAMA")
print("============================================================")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Model:", model_id)
print("Preparing HuggingFace pipeline...")
print("This may take some time...\n")


try:

    pipe = pipeline(
        task="text-generation",
        model=model_id,
        tokenizer=model_id,
        device=-1,
        max_new_tokens=50,
        do_sample=False,
        return_full_text=False
    )

    print("\nTinyLlama pipeline loaded successfully.")

except Exception as e:

    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR WHILE LOADING TINYLLAMA")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(type(e).__name__)
    print(str(e))

    raise


# ============================================================
# 7. CONVERT TO LANGCHAIN LLM
# ============================================================

print("\n============================================================")
print("[5/6] CREATING LANGCHAIN LLM")
print("============================================================")

llm_pipeline = HuggingFacePipeline(
    pipeline=pipe
)

llm = ChatHuggingFace(
    llm=llm_pipeline
)

print("LangChain LLM created successfully.")


# ============================================================
# 8. TEST TINYLLAMA FIRST
# ============================================================

print("\n============================================================")
print("TESTING TINYLLAMA")
print("============================================================")

try:

    test_response = llm.invoke(
        "Say hello in one short sentence."
    )

    print("\nTinyLlama test response:")
    print(test_response)

except Exception as e:

    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR WHILE RUNNING TINYLLAMA")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(type(e).__name__)
    print(str(e))

    raise


# ============================================================
# 9. CREATE MULTI QUERY RETRIEVER
# ============================================================

print("\n============================================================")
print("[6/6] CREATING MULTIQUERY RETRIEVER")
print("============================================================")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    ),
    llm=llm
)

print("MultiQueryRetriever created successfully.")


# ============================================================
# 10. QUERY
# ============================================================

query = "How to improve energy levels and maintain balance?"

print("\n============================================================")
print("QUERY")
print("============================================================")

print(query)


# ============================================================
# 11. NORMAL SIMILARITY SEARCH
# ============================================================

print("\n============================================================")
print("RUNNING NORMAL SIMILARITY SEARCH")
print("============================================================")

similarity_result = similarity_retriever.invoke(query)

print(
    "Similarity results found:",
    len(similarity_result)
)


# ============================================================
# 12. DISPLAY SIMILARITY RESULTS
# ============================================================

print("\n============================================================")
print("SIMILARITY SEARCH RESULTS")
print("============================================================")

for i, doc in enumerate(similarity_result):

    print(f"\n----- Result {i + 1} -----")

    print("Content:")
    print(doc.page_content)

    print("Source:")
    print(doc.metadata.get("source"))


# ============================================================
# 13. MULTI QUERY RETRIEVAL
# ============================================================

print("\n============================================================")
print("RUNNING MULTI-QUERY RETRIEVER")
print("============================================================")

print(
    "\nTinyLlama will generate alternative versions "
    "of the query."
)

print("Please wait...\n")


try:

    multiquery_result = multi_query_retriever.invoke(query)

except Exception as e:

    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR WHILE RUNNING MULTIQUERY RETRIEVER")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(type(e).__name__)
    print(str(e))

    raise


# ============================================================
# 14. DISPLAY MULTI QUERY RESULTS
# ============================================================

print("\n============================================================")
print("MULTI-QUERY RESULTS")
print("============================================================")

print(
    "Multi-query results found:",
    len(multiquery_result)
)


for i, doc in enumerate(multiquery_result):

    print(f"\n----- Result {i + 1} -----")

    print("Content:")
    print(doc.page_content)

    print("Source:")
    print(doc.metadata.get("source"))


# ============================================================
# 15. DONE
# ============================================================

print("\n============================================================")
print("PROGRAM FINISHED")
print("============================================================")