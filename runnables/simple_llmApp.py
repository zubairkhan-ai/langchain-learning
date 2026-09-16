from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. LOCAL HUGGING FACE MODEL
# ============================================================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,

    max_new_tokens=50,
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
# 3. CHAT HUGGING FACE MODEL
# ============================================================

model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# 4. PROMPT TEMPLATE
# ============================================================

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)


# ============================================================
# 5. DEFINE THE INPUT
# ============================================================

topic = input("Enter a topic: ")


# ============================================================
# 6. FORMAT THE PROMPT
# ============================================================

formatted_prompt = prompt.invoke({
    "topic": topic
})


# ============================================================
# 7. CALL THE LLM DIRECTLY
# ============================================================

result = model.invoke(formatted_prompt)


# ============================================================
# 8. PRINT THE OUTPUT
# ============================================================

print("\n==============================")
print("GENERATED BLOG TITLE")
print("==============================")

print(result.content)

