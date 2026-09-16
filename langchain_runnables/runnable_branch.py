from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch


# --------------------------------
# 1. Load TinyLlama
# --------------------------------

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=50,
    do_sample=False,
    return_full_text=False
)


# --------------------------------
# 2. Create LangChain Model
# --------------------------------

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# --------------------------------
# 3. Prompt for Classification
# --------------------------------

prompt = PromptTemplate(
    template="""
Classify the following customer email into exactly ONE category.

Categories:
- complaint
- refund
- general

Return ONLY one word from these categories.

Email:
{email}
""",
    input_variables=["email"]
)


# --------------------------------
# 4. Parser
# --------------------------------

parser = StrOutputParser()


# --------------------------------
# 5. Classification Chain
# --------------------------------

classifier_chain = prompt | model | parser


# --------------------------------
# 6. Branch Functions
# --------------------------------

def is_complaint(result):
    return "complaint" in result.lower()


def is_refund(result):
    return "refund" in result.lower()


# --------------------------------
# 7. Complaint Chain
# --------------------------------

complaint_prompt = PromptTemplate(
    template="""
The customer has made a complaint.

Customer message:
{email}

Write a polite customer-service response to the complaint.
""",
    input_variables=["email"]
)

complaint_chain = complaint_prompt | model | parser


# --------------------------------
# 8. Refund Chain
# --------------------------------

refund_prompt = PromptTemplate(
    template="""
The customer is asking for a refund.

Customer message:
{email}

Write a polite response saying that the refund request
has been received and will be processed.
""",
    input_variables=["email"]
)

refund_chain = refund_prompt | model | parser


# --------------------------------
# 9. General Chain
# --------------------------------

general_prompt = PromptTemplate(
    template="""
The customer has a general question.

Customer message:
{email}

Write a helpful response to the customer.
""",
    input_variables=["email"]
)

general_chain = general_prompt | model | parser


# --------------------------------
# 10. Runnable Branch
# --------------------------------

branch = RunnableBranch(
    (is_complaint, complaint_chain),
    (is_refund, refund_chain),
    general_chain
)


# --------------------------------
# 11. Complete Chain
# --------------------------------

final_chain = classifier_chain | branch


# --------------------------------
# 12. Test
# --------------------------------

email = """
I am very disappointed with your service.
My order arrived late and the product was damaged.
"""

result = final_chain.invoke({
    "email": email
})


print("FINAL RESPONSE:")
print(result)