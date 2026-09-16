from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. CREATE DOCUMENTS
# ============================================================

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and excellent chasing ability.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm leadership and powerful batting.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicket-keeping, and leadership are highly respected.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his accurate yorkers and variations.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and powerful batting make him a valuable player.",
    metadata={"team": "Chennai Super Kings"}
)


docs = [doc1, doc2, doc3, doc4, doc5]


# ============================================================
# 3. CREATE CHROMA VECTOR STORE
# ============================================================

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_name="sample"
)


# ============================================================
# 4. ADD DOCUMENTS
# ============================================================

ids = vector_store.add_documents(docs)

print("Document IDs:")
print(ids)


# ============================================================
# 5. GET STORED DOCUMENTS
# ============================================================

result = vector_store.get(
    include=["embeddings", "documents", "metadatas"]
)

print("\nStored Documents:")
print(result)


# ============================================================
# 6. SIMILARITY SEARCH
# ============================================================

results = vector_store.similarity_search(
    query="who among these are bowler",
    k=2
)

print("\nSimilarity Search:")
for doc in results:
    print("\nContent:", doc.page_content)
    print("Metadata:", doc.metadata)


# ============================================================
# 7. SIMILARITY SEARCH WITH SCORES
# ============================================================

results_with_scores = vector_store.similarity_search_with_score(
    query="who among these are bowler",
    k=2
)

print("\nSimilarity Search With Scores:")

for doc, score in results_with_scores:
    print("\nContent:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("Score:", score)


# ============================================================
# 8. METADATA FILTER
# ============================================================

results = vector_store.similarity_search(
    query="",
    k=5,
    filter={"team": "Chennai Super Kings"}
)

print("\nChennai Super Kings Documents:")

for doc in results:
    print("\nContent:", doc.page_content)
    print("Metadata:", doc.metadata)


# ============================================================
# 9. UPDATE A DOCUMENT
# ============================================================

updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances.",
    metadata={
        "team": "Royal Challengers Bangalore"
    }
)


# Use the ID returned by add_documents()
vector_store.update_document(
    document_id=ids[0],
    document=updated_doc1
)


# ============================================================
# 10. CHECK UPDATED DOCUMENT
# ============================================================

result = vector_store.get(
    include=["embeddings", "documents", "metadatas"]
)

print("\nAfter Update:")
print(result)