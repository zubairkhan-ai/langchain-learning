# ============================================================
# LOCAL TOOL CALLING WITHOUT OPENAI API KEY
# TinyLlama + LangChain Tool Calling
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from transformers import pipeline

from langchain_core.tools import tool

from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)


# ============================================================
# 2. CREATE CUSTOM TOOL
# ============================================================

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers and return their product.
    """
    return a * b


# ============================================================
# 3. LOAD TINYLLAMA
# ============================================================

print("\n============================================================")
print("LOADING TINYLLAMA")
print("============================================================")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    task="text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=150,
    do_sample=False,
    return_full_text=False,
    clean_up_tokenization_spaces=False
)

print("TinyLlama loaded successfully.")


# ============================================================
# 4. CREATE LANGCHAIN MODEL
# ============================================================

llm_pipeline = HuggingFacePipeline(
    pipeline=pipe
)

llm = ChatHuggingFace(
    llm=llm_pipeline
)

print("LangChain model created successfully.")


# ============================================================
# 5. BIND TOOL TO MODEL
# ============================================================

llm_with_tools = llm.bind_tools(
    tools=[multiply]
)

print("Tool bound successfully.")


# ============================================================
# 6. DISPLAY TOOL DETAILS
# ============================================================

print("\n============================================================")
print("TOOL INFORMATION")
print("============================================================")

print("Tool name:", multiply.name)

print("Tool description:", multiply.description)

print("Tool arguments:", multiply.args)


# ============================================================
# 7. ASK THE MODEL
# ============================================================

print("\n============================================================")
print("ASKING THE MODEL")
print("============================================================")

question = "Can you multiply 3 with 10?"

result = llm_with_tools.invoke(
    [
        HumanMessage(content=question)
    ]
)

print("User question:", question)

print("\nModel response:")
print(result.content)

print("\nGenerated tool calls:")
print(result.tool_calls)


# ============================================================
# 8. CHECK WHETHER MODEL GENERATED A TOOL CALL
# ============================================================

if result.tool_calls:

    print("\n============================================================")
    print("STRUCTURED TOOL CALL FOUND")
    print("============================================================")

    tool_call = result.tool_calls[0]

    print("Complete tool call:")
    print(tool_call)

    print("\nTool name:")
    print(tool_call["name"])

    print("\nTool arguments:")
    print(tool_call["args"])

    print("\nTool call ID:")
    print(tool_call["id"])

    # ========================================================
    # 9. EXECUTE TOOL
    # ========================================================

    tool_result = multiply.invoke(
        tool_call["args"]
    )

else:

    print("\n============================================================")
    print("NO STRUCTURED TOOL CALL FOUND")
    print("============================================================")

    print(
        "TinyLlama returned normal text instead of a structured "
        "tool call."
    )

    print("\nCreating a tool call manually for demonstration...")

    tool_call = {
        "name": "multiply",
        "args": {
            "a": 3,
            "b": 10
        },
        "id": "manual_call_001",
        "type": "tool_call"
    }

    print("\nManual tool call:")
    print(tool_call)

    # ========================================================
    # 10. EXECUTE TOOL MANUALLY
    # ========================================================

    tool_result = multiply.invoke(
        tool_call["args"]
    )


# ============================================================
# 11. DISPLAY TOOL RESULT
# ============================================================

print("\n============================================================")
print("TOOL EXECUTION RESULT")
print("============================================================")

print("Tool result:", tool_result)


# ============================================================
# 12. CREATE TOOL MESSAGE
# ============================================================

tool_message = ToolMessage(
    content=str(tool_result),
    name=tool_call["name"],
    tool_call_id=tool_call["id"]
)

print("\n============================================================")
print("TOOL MESSAGE")
print("============================================================")

print(tool_message)


# ============================================================
# 13. SHOW COMPLETE MESSAGE FLOW
# ============================================================

print("\n============================================================")
print("COMPLETE TOOL CALLING FLOW")
print("============================================================")

messages = [
    HumanMessage(content=question),
    result,
    tool_message
]

for index, message in enumerate(messages, start=1):

    print(f"\n----- Message {index} -----")

    print("Message type:", type(message).__name__)

    print("Message content:", message.content)

    if hasattr(message, "tool_calls"):

        print("Tool calls:", message.tool_calls)

    if hasattr(message, "tool_call_id"):

        print("Tool call ID:", message.tool_call_id)