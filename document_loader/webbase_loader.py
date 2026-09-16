from langchain_community.document_loaders import WebBaseLoader


# Website load karo
loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/Cricket"
)


# Website se documents load karo
docs = loader.load()


# Documents print karo
print("Number of documents:", len(docs))

print("\nCONTENT:")
print(docs[0].page_content[:2000])

print("\nMETADATA:")
print(docs[0].metadata)
print(docs[0].page_content)