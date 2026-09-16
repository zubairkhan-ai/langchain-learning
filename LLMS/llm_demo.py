from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm=OpenAI(model='gpt-3.5-turbo-instruct')
llm.invoke("what is capital of pakistan")
result=llm.invoke("what is capital of pakistan")
print(result)