# ============================================================
# PDF RAG / PDF READER
# PDF -> Loader -> Splitter -> Embeddings -> Vector Store
# -> Retriever -> LLM -> Parser
# ============================================================


from transformers import pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. PDF LOADER
# ============================================================

pdf_path = "runnables/sample.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()


print("\n==============================")
print("PDF LOADER")
print("==============================")

print("Total pages:", len(documents))


# ============================================================
# 2. TEXT SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


print("\n==============================")
print("TEXT SPLITTER")
print("==============================")

print("Total chunks:", len(chunks))


# ============================================================
# 3. EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("\n==============================")
print("EMBEDDINGS")
print("==============================")

print("Embedding model loaded")


# ============================================================
# 4. VECTOR STORE
# ============================================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


print("\n==============================")
print("VECTOR STORE")
print("==============================")

print("FAISS vector store created")


# ============================================================
# 5. RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


print("\n==============================")
print("RETRIEVER")
print("==============================")

print("Retriever created")


# ============================================================
# 6. LOCAL TINYLLAMA
# ============================================================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,

    max_new_tokens=150,
    do_sample=False,

    return_full_text=False,
    clean_up_tokenization_spaces=False
)


llm = HuggingFacePipeline(
    pipeline=pipe
)


model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# 7. PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],

    template="""
You are a helpful assistant.

Answer the question using ONLY the information provided
in the context below.

If the answer is not present in the context,
say "The answer is not available in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 8. OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# 9. RAG CHAIN
# ============================================================

chain = prompt | model | parser


# ============================================================
# 10. ASK QUESTION
# ============================================================

question = input("\nAsk a question about the PDF: ")


# ============================================================
# 11. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

retrieved_docs = retriever.invoke(question)


print("\n==============================")
print("RETRIEVED DOCUMENTS")
print("==============================")

print("Documents retrieved:", len(retrieved_docs))


# ============================================================
# 12. CREATE CONTEXT
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in retrieved_docs
)


# ============================================================
# 13. SEND CONTEXT + QUESTION TO LLM
# ============================================================

result = chain.invoke({
    "context": context,
    "question": question
})


# ============================================================
# 14. FINAL ANSWER
# ============================================================

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(result)