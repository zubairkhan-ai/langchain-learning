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
    max_new_tokens=50,
    do_sample=False,
    return_full_text=False,
    clean_up_tokenization_spaces=False
)


# ==========================================
# HUGGING FACE MODEL
# ==========================================

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# ==========================================
# CHAT HISTORY
# ==========================================

chat_history = [
    SystemMessage(
        content="""You are a helpful and accurate AI assistant.
Answer questions directly.
For simple math and comparison questions, reason carefully before answering.
Never invent facts.
Keep answers short and clear."""
    )
] 


# ==========================================
# CHAT LOOP
# ==========================================

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break

    # Add user's message
    chat_history.append(
        HumanMessage(content=user_input)
    )

    # Send complete conversation
    result = model.invoke(chat_history)

    # Add AI response to history
    chat_history.append(
        AIMessage(content=result.content)
    )

    print("AI:", result.content)