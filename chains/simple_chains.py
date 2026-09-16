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

    max_new_tokens=150,
    do_sample=False,

    return_full_text=False,
    clean_up_tokenization_spaces=False
)


# ============================================================
# 2. HUGGING FACE PIPELINE
# ============================================================

llm = HuggingFacePipeline(
    pipeline=pipe
)


# ============================================================
# 3. CHAT HUGGING FACE
# ============================================================

model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# 4. PROMPT TEMPLATE
# ============================================================

prompt = PromptTemplate(
    template="""
Generate 5 interesting facts about {topic}.

Return the facts as a simple numbered list.
""",
    input_variables=["topic"]
)


# ============================================================
# 5. STRING OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# 6. LCEL CHAIN
# ============================================================

chain = prompt | model | parser


# ============================================================
# 7. INVOKE CHAIN
# ============================================================

result = chain.invoke({
    "topic": "cricket"
})


# ============================================================
# 8. OUTPUT
# ============================================================

print("\n==============================")
print("5 INTERESTING FACTS")
print("==============================")

print(result)

chain.get_graph().print_ascii()