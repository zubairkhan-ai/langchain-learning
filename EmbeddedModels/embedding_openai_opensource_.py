from langchain_huggingface import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text='dehli is the capital of india'
vector=embedding.aembed_query("text")
print(str(vector))