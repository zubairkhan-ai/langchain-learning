from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


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
# 2. Create LangChain model
# --------------------------------

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# --------------------------------
# 3. Create Tweet Prompt
# --------------------------------

tweet_prompt = PromptTemplate(
    template="""
Write a short and engaging tweet about {topic}.

Keep it concise and interesting.
Do not explain anything.
Only return the tweet.
""",
    input_variables=["topic"]
)


# --------------------------------
# 4. Create LinkedIn Prompt
# --------------------------------

linkedin_prompt = PromptTemplate(
    template="""
Write a professional LinkedIn post about {topic}.

Make it informative and engaging.
Use a professional tone.
Only return the LinkedIn post.
""",
    input_variables=["topic"]
)


# --------------------------------
# 5. Output Parser
# --------------------------------

parser = StrOutputParser()


# --------------------------------
# 6. Create two chains
# --------------------------------

tweet_chain = tweet_prompt | model | parser

linkedin_chain = linkedin_prompt | model | parser


# --------------------------------
# 7. Run them in parallel
# --------------------------------

parallel_chain = RunnableParallel(
    tweet=tweet_chain,
    linkedin=linkedin_chain
)


# --------------------------------
# 8. Invoke
# --------------------------------

result = parallel_chain.invoke({
    "topic": "Artificial Intelligence"
})


# --------------------------------
# 9. Print results
# --------------------------------

print("TWEET:")
print(result["tweet"])

print("\nLINKEDIN POST:")
print(result["linkedin"])