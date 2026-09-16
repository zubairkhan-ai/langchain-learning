from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_community.document_loaders import TextLoader

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate


# --------------------------------
# 1. Load TinyLlama
# --------------------------------

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=100,
    do_sample=False,
    return_full_text=False
)


# --------------------------------
# 2. Create LangChain Model
# --------------------------------

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# --------------------------------
# 3. Create Prompt
# --------------------------------

prompt = PromptTemplate(
    template="""Write a short summary for the following poem:

{poem}

Only return the summary.
""",
    input_variables=["poem"]
)


# --------------------------------
# 4. Output Parser
# --------------------------------

parser = StrOutputParser()


# --------------------------------
# 5. Load Text File
# --------------------------------

loader = TextLoader(
    "document_loader/cricket_poem.txt",
    encoding="utf-8"
)

docs = loader.load()


# --------------------------------
# 6. Print loaded documents
# --------------------------------

print("DOCUMENT:")
print(docs)

print("\nDOCUMENT CONTENT:")
print(docs[0].page_content)


# --------------------------------
# 7. Create Chain
# --------------------------------

chain = prompt | model | parser


# --------------------------------
# 8. Invoke Chain
# --------------------------------

result = chain.invoke({
    "poem": docs[0].page_content
})


# --------------------------------
# 9. Print Summary
# --------------------------------

print("\nSUMMARY:")
print(result)