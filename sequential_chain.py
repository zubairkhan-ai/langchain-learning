from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. LOCAL HUGGING FACE MODEL
# ============================================================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,

    max_new_tokens=100,
    do_sample=False,

    return_full_text=False,
    clean_up_tokenization_spaces=False
)


# ============================================================
# 2. HUGGING FACE LLM
# ============================================================

llm = HuggingFacePipeline(
    pipeline=pipe
)


# ============================================================
# 3. CHAT MODEL
# ============================================================

model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# 4. FIRST PROMPT
# ============================================================

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)


# ============================================================
# 5. SECOND PROMPT
# ============================================================

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n\n{text}",
    input_variables=["text"]
)


# ============================================================
# 6. OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# 7. SEQUENTIAL CHAIN
# ============================================================

chain = prompt1 | model | parser | prompt2 | model | parser


# ============================================================
# 8. INVOKE
# ============================================================

result = chain.invoke({
    "topic": "unemployement"
})


# ============================================================
# 9. FINAL OUTPUT
# ============================================================

print("\n==============================")
print("5 POINTER SUMMARY")
print("==============================")

print(result)


# ============================================================
# 10. GRAPH
# ============================================================

print("\n==============================")
print("CHAIN GRAPH")
print("==============================")

chain.get_graph().print_ascii()