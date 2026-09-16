from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# LOCAL TINYLLAMA
# ==========================================

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


# ==========================================
# HUGGING FACE PIPELINE
# ==========================================

llm = HuggingFacePipeline(
    pipeline=pipe
)


# ==========================================
# CHAT HUGGING FACE
# ==========================================

model = ChatHuggingFace(
    llm=llm
)


# ==========================================
# STRING OUTPUT PARSER
# ==========================================

parser = StrOutputParser()


# ==========================================
# FIRST PROMPT TEMPLATE
# ==========================================

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)


# ==========================================
# SECOND PROMPT TEMPLATE
# ==========================================

template2 = PromptTemplate(
    template="""Summarize the following text in EXACTLY 5 short lines.

Do not write an introduction.
Do not explain anything.
Do not number the lines.
Return exactly 5 lines.

Text:
{text}""",
    input_variables=["text"]
)


# ==========================================
# FIRST PROMPT
# ==========================================

prompt1 = template1.invoke({
    "topic": "black hole"
})


# ==========================================
# FIRST MODEL CALL
# ==========================================

result = model.invoke(prompt1)


# ==========================================
# STRING PARSER
# ==========================================

report = parser.invoke(result)


print("\n==============================")
print("DETAILED REPORT")
print("==============================")

print(report)


# ==========================================
# SECOND PROMPT
# ==========================================

prompt2 = template2.invoke({
    "text": report
})


# ==========================================
# SECOND MODEL CALL
# ==========================================

result1 = model.invoke(prompt2)


# ==========================================
# STRING PARSER
# ==========================================

summary = parser.invoke(result1)


print("\n==============================")
print("5 LINE SUMMARY")
print("==============================")

print(summary)