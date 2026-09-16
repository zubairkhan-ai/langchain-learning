from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


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
# 4. OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# 5. NOTES PROMPT
# ============================================================

notes_prompt = PromptTemplate(
    template="""
Create short and easy-to-understand study notes about {topic}.

Include:
- Definition
- Important concepts
- Key points
- One simple example

Keep the notes concise and useful for exam preparation.
""",
    input_variables=["topic"]
)


# ============================================================
# 6. QUIZ PROMPT
# ============================================================

quiz_prompt = PromptTemplate(
    template="""
Create a short quiz about {topic}.

Generate:
1. Three multiple-choice questions.
2. Give four options for each question.
3. Clearly mention the correct answer.

Keep the questions simple and useful for students.
""",
    input_variables=["topic"]
)


# ============================================================
# 7. NOTES CHAIN
# ============================================================

notes_chain = notes_prompt | model | parser


# ============================================================
# 8. QUIZ CHAIN
# ============================================================

quiz_chain = quiz_prompt | model | parser


# ============================================================
# 9. PARALLEL CHAIN
# ============================================================

parallel_chain = RunnableParallel(
    notes=notes_chain,
    quiz=quiz_chain
)


# ============================================================
# 10. INVOKE PARALLEL CHAIN
# ============================================================

result = parallel_chain.invoke({
    "topic": "Computer Networks"
})


# ============================================================
# 11. NOTES OUTPUT
# ============================================================

print("\n==============================")
print("STUDY NOTES")
print("==============================")

print(result["notes"])


# ============================================================
# 12. QUIZ OUTPUT
# ============================================================

print("\n==============================")
print("QUIZ")
print("==============================")

print(result["quiz"])


# ============================================================
# 13. SHOW CHAIN GRAPH
# ============================================================

print("\n==============================")
print("CHAIN GRAPH")
print("==============================")

parallel_chain.get_graph().print_ascii()