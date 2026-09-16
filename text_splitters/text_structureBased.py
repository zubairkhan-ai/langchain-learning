from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# 1. Get PDF path
# --------------------------------

pdf_path = Path(__file__).parent / "cricket_6_page.pdf"


# --------------------------------
# 2. Load PDF
# --------------------------------

loader = PyPDFLoader(str(pdf_path))

docs = loader.load()

print("Number of pages:", len(docs))


# --------------------------------
# 3. Create Recursive Text Splitter
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# --------------------------------
# 4. Split Documents
# --------------------------------

chunks = splitter.split_documents(docs)


# --------------------------------
# 5. Print number of chunks
# --------------------------------

print("Number of chunks:", len(chunks))


# --------------------------------
# 6. Print chunks
# --------------------------------

for i, chunk in enumerate(chunks):

    print(f"\n========== CHUNK {i + 1} ==========")

    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)