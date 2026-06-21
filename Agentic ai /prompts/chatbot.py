from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt   

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content='you are a helpful  AI assisstant')
]

while True:
    user_input= input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if(user_input == 'exit'):
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))  
    print("AI: ",result.content)
print(chat_history)
