from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


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
# HUGGING FACE + LANGCHAIN
# ==========================================

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# ==========================================
# MESSAGES
# ==========================================

messages = [

    SystemMessage(
        content="You are a helpful assistant."
    ),

    HumanMessage(
        content="Tell me about LangChain."
    )

]


# ==========================================
# INVOKE MODEL
# ==========================================

result = model.invoke(messages)


# ==========================================
# ADD AI RESPONSE
# ==========================================

messages.append(
    AIMessage(
        content=result.content
    )
)


# ==========================================
# PRINT
# ==========================================

print(messages)