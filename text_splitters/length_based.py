from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# PDF path
pdf_path = Path(__file__).parent / "cricket_6_page.pdf"


# Load PDF
loader = PyPDFLoader(str(pdf_path))

docs = loader.load()


print("Number of pages:", len(docs))


# Text splitter
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)


# Split documents
result = splitter.split_documents(docs)


print("\nNumber of chunks:", len(result))


for i, doc in enumerate(result):
    print(f"\n--- CHUNK {i + 1} ---")
    print(doc.page_content)
print(result[0].page_content)