from langchain_core.tools import tool


# ============================================================
# CREATE CUSTOM TOOL
# ============================================================

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b


# ============================================================
# DISPLAY TOOL INFORMATION
# ============================================================

print("Tool name:", multiply.name)
print("Tool description:", multiply.description)
print("tools arg",multiply.args)


# ============================================================
# INVOKE THE TOOL
# ============================================================

result = multiply.invoke({
    "a": 3,
    "b": 5
})


# ============================================================
# PRINT RESULT
# ============================================================

print("\n================ TOOL RESULT ================\n")
print("3 × 5 =", result)
print(multiply.args_schema.model_json_schema())