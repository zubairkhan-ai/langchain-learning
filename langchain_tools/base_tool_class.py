# ============================================================
# CUSTOM TOOL USING BASETOOL
# ============================================================

from typing import Type

from langchain_core.tools import BaseTool

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
# 2. CREATE CUSTOM BASETOOL CLASS
# ============================================================

class MultiplyTool(BaseTool):

    name: str = "multiply"

    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    # ========================================================
    # 3. TOOL LOGIC
    # ========================================================

    def _run(self, a: int, b: int) -> int:
        """
        Multiply two numbers.
        """
        return a * b


# ============================================================
# 4. CREATE TOOL OBJECT
# ============================================================

multiply_tool = MultiplyTool()


# ============================================================
# 5. INVOKE TOOL
# ============================================================

result = multiply_tool.invoke({
    "a": 3,
    "b": 3
})


# ============================================================
# 6. PRINT OUTPUT
# ============================================================

print("\n================ TOOL RESULT ================\n")

print("Result:", result)

print("\nTool name:", multiply_tool.name)

print("\nTool description:", multiply_tool.description)

print("\nTool arguments:")

print(multiply_tool.args)