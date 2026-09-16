from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


# -------------------------
# 1. Load TinyLlama
# -------------------------

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=100,
    do_sample=True,
    return_full_text=False
)


# -------------------------
# 2. Convert HF pipeline
#    into LangChain model
# -------------------------

llm = HuggingFacePipeline(
    pipeline=pipe
)


# -------------------------
# 3. Convert it to Chat model
# -------------------------

model = ChatHuggingFace(
    llm=llm
)


# -------------------------
# 4. Prompt
# -------------------------

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template='explain the following -{text}',
    input_variables=['text']
)

# -------------------------
# 5. Output parser
# -------------------------

parser = StrOutputParser()


# -------------------------
# 6. Create RunnableSequence
# -------------------------

chain = RunnableSequence(
    prompt,
    model,
    parser,prompt2,model,parser
)


# -------------------------
# 7. Run chain
# -------------------------

result = chain.invoke({
    "topic": "AI"
})

print(result)