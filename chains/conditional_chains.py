from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch


# ============================================================
# 1. LOCAL HUGGING FACE MODEL
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
# 5. POSITIVE FEEDBACK PROMPT
# ============================================================

positive_prompt = PromptTemplate(
    template="""
The customer has given positive feedback.

Customer feedback:
{feedback}

Write a short, friendly and thankful response.
Thank the customer for their positive feedback.
Do not apologize.
""",
    input_variables=["feedback"]
)


# ============================================================
# 6. NEGATIVE FEEDBACK PROMPT
# ============================================================

negative_prompt = PromptTemplate(
    template="""
The customer has given negative feedback.

Customer feedback:
{feedback}

Write a short, polite response.
Apologize for the bad experience.
Thank the customer for the feedback.
Say that the feedback will help improve the service.
""",
    input_variables=["feedback"]
)


# ============================================================
# 7. POSITIVE CHAIN
# ============================================================

positive_chain = positive_prompt | model | parser


# ============================================================
# 8. NEGATIVE CHAIN
# ============================================================

negative_chain = negative_prompt | model | parser


# ============================================================
# 9. CONDITION FUNCTION
# ============================================================

def check_feedback(data):

    feedback = data["feedback"].lower()

    positive_words = [
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "love",
        "helpful",
        "nice",
        "perfect",
        "awesome",
        "fantastic",
        "best",
        "happy",
        "satisfied"
    ]

    negative_words = [
        "bad",
        "poor",
        "terrible",
        "awful",
        "hate",
        "worst",
        "disappointed",
        "disappointing",
        "useless",
        "slow",
        "broken",
        "problem",
        "issue",
        "waste"
    ]

    # Check negative feedback first
    for word in negative_words:

        if word in feedback:
            return False

    # Check positive feedback
    for word in positive_words:

        if word in feedback:
            return True

    # Default: negative chain
    return False


# ============================================================
# 10. CONDITIONAL CHAIN
# ============================================================

conditional_chain = RunnableBranch(

    # If check_feedback() returns True
    (check_feedback, positive_chain),

    # Otherwise
    negative_chain
)


# ============================================================
# 11. USER INPUT
# ============================================================

feedback = input("Enter your feedback: ")


# ============================================================
# 12. INVOKE CONDITIONAL CHAIN
# ============================================================

result = conditional_chain.invoke({
    "feedback": feedback
})


# ============================================================
# 13. DISPLAY RESULT
# ============================================================

print("\n==============================")
print("CUSTOMER FEEDBACK")
print("==============================")

print(feedback)


print("\n==============================")
print("RESPONSE")
print("==============================")

print(result)


# ============================================================
# 14. SHOW GRAPH
# ============================================================

print("\n==============================")
print("CHAIN GRAPH")
print("==============================")

conditional_chain.get_graph().print_ascii()