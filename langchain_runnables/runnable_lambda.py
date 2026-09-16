from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
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
    do_sample=False,
    return_full_text=False
)


# --------------------------------
# 2. Create LangChain model
# --------------------------------

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# --------------------------------
# 3. Joke Prompt
# --------------------------------

prompt = PromptTemplate(
    template="""
Write a short and funny joke about {topic}.

Only return the joke.
""",
    input_variables=["topic"]
)


# --------------------------------
# 4. Output Parser
# --------------------------------

parser = StrOutputParser()


# --------------------------------
# 5. Joke Generation Chain
# --------------------------------

joke_chain = prompt | model | parser


# --------------------------------
# 6. Word Count Function
# --------------------------------

def count_words(joke):
    return len(joke.split())


# --------------------------------
# 7. Convert function to Runnable
# --------------------------------

word_count_chain = RunnableLambda(count_words)


# --------------------------------
# 8. Runnable Parallel
# --------------------------------

parallel_chain = RunnableParallel(
    joke=RunnablePassthrough(),
    word_count=word_count_chain
)


# --------------------------------
# 9. Complete Chain
# --------------------------------

final_chain = joke_chain | parallel_chain


# --------------------------------
# 10. Invoke
# --------------------------------

result = final_chain.invoke({
    "topic": "wedding"
})


# --------------------------------
# 11. Print results
# --------------------------------

print("JOKE:")
print(result["joke"])

print("\nWORD COUNT:")
print(result["word_count"])