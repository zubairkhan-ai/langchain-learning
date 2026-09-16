from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path(__file__).parent / "cricket_6_page.pdf"

loader = PyPDFLoader(str(pdf_path))

docs = loader.load()

print("Number of pages:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n--- PAGE {i + 1} ---")
    print(doc.page_content[:500])
print(docs[0].page_content)
print(docs[0].metadata)