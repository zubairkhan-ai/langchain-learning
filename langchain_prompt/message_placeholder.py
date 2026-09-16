from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage
chat_template = ChatPromptTemplate.from_messages([

    ('system', 'you are helpful customer support agent'),

    MessagesPlaceholder(variable_name='chat_history'),

    ('human', '{query}')

])


chat_history = []

with open('langchain_prompt/chat_history.txt', 'r', encoding='utf-8') as f:

    chat_history.extend(f.readlines())


print(chat_history)
prompt=chat_template.invoke({'chat_history':chat_history,'query':HumanMessage(content='where is my refund')})
print(prompt)