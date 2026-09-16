from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader


# -------------------------
# Load TXT files
# -------------------------

txt_loader = DirectoryLoader(
    "document_loader",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

txt_docs = txt_loader.load()


# -------------------------
# Load PDF files
# -------------------------

pdf_loader = DirectoryLoader(
    "document_loader",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

pdf_docs = pdf_loader.load()


# -------------------------
# Combine documents
# -------------------------

docs = txt_docs + pdf_docs


print("Total documents:", len(docs))


# -------------------------
# Show documents
# -------------------------

for i, doc in enumerate(docs):
    print(f"\n========== DOCUMENT {i + 1} ==========")
    print(doc.page_content[:500])
    print("SOURCE:", doc.metadata.get("source"))