from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)


# --------------------------------
# 1. Load TinyLlama
# --------------------------------

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=100,
    do_sample=True,
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
# 3. Joke Generator Prompt
# --------------------------------

joke_prompt = PromptTemplate(
    template="""
Write a short and funny joke about {topic}.

Only return the joke.
""",
    input_variables=["topic"]
)


# --------------------------------
# 4. Explanation Prompt
# --------------------------------

explanation_prompt = PromptTemplate(
    template="""
Explain the following joke in simple words:

{joke}

Only return the explanation.
""",
    input_variables=["joke"]
)


# --------------------------------
# 5. Output Parser
# --------------------------------

parser = StrOutputParser()


# --------------------------------
# 6. First Chain: Generate Joke
# --------------------------------

joke_chain = joke_prompt | model | parser


# --------------------------------
# 7. Second Chain: Generate Explanation
# --------------------------------

explanation_chain = explanation_prompt | model | parser


# --------------------------------
# 8. Runnable Parallel
# --------------------------------

parallel_chain = RunnableParallel(
    joke=RunnablePassthrough(),
    explanation=explanation_chain
)


# --------------------------------
# 9. Complete Chain
# --------------------------------

final_chain = joke_chain | parallel_chain


# --------------------------------
# 10. Invoke
# --------------------------------

result = final_chain.invoke({
    "topic": "Artificial Intelligence"
})


# --------------------------------
# 11. Print
# --------------------------------

print("JOKE:")
print(result["joke"])

print("\nEXPLANATION:")
print(result["explanation"])