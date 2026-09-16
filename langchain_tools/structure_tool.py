# ============================================================
# STRUCTURED TOOL EXAMPLE
# ============================================================

from langchain_core.tools import StructuredTool

from pydantic import BaseModel, Field


# ============================================================
# 1. INPUT SCHEMA
# ============================================================

class MultiplyInput(BaseModel):

    a: int = Field(
        ...,
        description="The first number to multiply"
    )

    b: int = Field(
        ...,
        description="The second number to multiply"
    )


# ============================================================
# 2. NORMAL PYTHON FUNCTION
# ============================================================

def multiply_func(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b


# ============================================================
# 3. CREATE STRUCTURED TOOL
# ============================================================

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)


# ============================================================
# 4. INVOKE TOOL
# ============================================================

result = multiply_tool.invoke({
    "a": 3,
    "b": 3
})


# ============================================================
# 5. PRINT RESULT
# ============================================================

print("\n================ TOOL RESULT ================\n")

print("Result:", result)

print("\nTool name:", multiply_tool.name)

print("\nTool description:", multiply_tool.description)

print("\nTool arguments schema:")

print(multiply_tool.args)