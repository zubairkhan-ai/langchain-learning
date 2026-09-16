from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)

from pydantic import BaseModel, Field
from typing import Literal
import json


# ============================================================
# 1. LOCAL TINYLLAMA
# ============================================================

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

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# 2. PYDANTIC SCHEMA
# ============================================================

class Student(BaseModel):

    name: str = Field(
        description="Student's name"
    )

    age: int = Field(
        gt=0,
        lt=100,
        description="Student's age must be between 1 and 99"
    )

    department: str = Field(
        description="Student's department"
    )

    grade: Literal["A", "B", "C", "D", "F"] = Field(
        description="Student's grade"
    )


# ============================================================
# 3. SHOW JSON SCHEMA
# ============================================================

schema = Student.model_json_schema()

print("\n==============================")
print("PYDANTIC JSON SCHEMA")
print("==============================")

print(json.dumps(schema, indent=4))


# ============================================================
# 4. STRUCTURED OUTPUT PARSER
# ============================================================

response_schemas = [

    ResponseSchema(
        name="name",
        description="Student's name"
    ),

    ResponseSchema(
        name="age",
        description="Student's age must be between 1 and 99"
    ),

    ResponseSchema(
        name="department",
        description="Student's department"
    ),

    ResponseSchema(
        name="grade",
        description="Student's grade must be A, B, C, D or F"
    )
]


parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)


# ============================================================
# 5. FORMAT INSTRUCTIONS
# ============================================================

format_instructions = parser.get_format_instructions()


print("\n==============================")
print("FORMAT INSTRUCTIONS")
print("==============================")

print(format_instructions)


# ============================================================
# 6. PROMPT
# ============================================================

template = PromptTemplate(
    template="""

You are a student information assistant.

Create information for this student:

Name: Ali
Age: 15
Department: Computer Science
Grade: A

Your task is to return the information using EXACTLY
the required JSON structure.

{format_instructions}

IMPORTANT RULES:

- Return ONLY JSON.
- Do NOT write explanations.
- Do NOT use markdown.
- Do NOT use ```json.
- Do NOT add extra fields.
- age must be an integer.
- grade must be one of A, B, C, D, F.

""",
    input_variables=[],
    partial_variables={
        "format_instructions": format_instructions
    }
)


# ============================================================
# 7. CREATE PROMPT
# ============================================================

prompt = template.invoke({})


# ============================================================
# 8. MODEL CALL
# ============================================================

result = model.invoke(prompt)


print("\n==============================")
print("RAW MODEL RESPONSE")
print("==============================")

print(result.content)


# ============================================================
# 9. STRUCTURED OUTPUT PARSER
# ============================================================

try:

    parsed_result = parser.parse(
        result.content
    )

    print("\n==============================")
    print("STRUCTURED OUTPUT")
    print("==============================")

    print(parsed_result)


    # ========================================================
    # 10. PYDANTIC VALIDATION
    # ========================================================

    student = Student(
        **parsed_result
    )


    print("\n==============================")
    print("PYDANTIC VALIDATED OUTPUT")
    print("==============================")

    print(student)


    # ========================================================
    # 11. INDIVIDUAL VALUES
    # ========================================================

    print("\n==============================")
    print("STUDENT INFORMATION")
    print("==============================")

    print("Name:", student.name)
    print("Age:", student.age)
    print("Department:", student.department)
    print("Grade:", student.grade)


except Exception as e:

    print("\n==============================")
    print("VALIDATION / PARSER ERROR")
    print("==============================")

    print(e)