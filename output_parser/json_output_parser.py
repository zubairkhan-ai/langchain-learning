from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ==========================================
# LOCAL TINYLLAMA
# ==========================================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,

    # Keep output short
    max_new_tokens=50,

    # Deterministic output
    do_sample=False,

    # Don't include the original prompt in output
    return_full_text=False,

    clean_up_tokenization_spaces=False
)


# ==========================================
# HUGGINGFACE PIPELINE
# ==========================================

llm = HuggingFacePipeline(
    pipeline=pipe
)


# ==========================================
# CHAT HUGGINGFACE
# ==========================================

model = ChatHuggingFace(
    llm=llm
)


# ==========================================
# JSON OUTPUT PARSER
# ==========================================

parser = JsonOutputParser()


# ==========================================
# PROMPT TEMPLATE
# ==========================================

template = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the following question:

Question:
What is the largest planet in our solar system?

Return ONLY a valid JSON object.

The JSON must contain exactly one key:

"answer"

Your response must look like this:

{{
    "answer": "your answer"
}}

IMPORTANT:
- Generate the answer yourself.
- Do not copy the example answer.
- Do not write any explanation.
- Do not use markdown.
- Do not use ``` .
- Return JSON only.
""",
    input_variables=[]
)


# ==========================================
# CREATE PROMPT
# ==========================================

prompt = template.invoke({})


# ==========================================
# CALL TINYLLAMA
# ==========================================

result = model.invoke(prompt)


# ==========================================
# RAW RESPONSE
# ==========================================

print("\n==============================")
print("RAW RESPONSE")
print("==============================")

print(result.content)


# ==========================================
# JSON OUTPUT PARSER
# ==========================================

try:

    parsed_result = parser.invoke(result)

    print("\n==============================")
    print("PARSED OUTPUT")
    print("==============================")

    print(parsed_result)

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(parsed_result["answer"])


except Exception as e:

    print("\n==============================")
    print("PARSER ERROR")
    print("==============================")

    print(e)