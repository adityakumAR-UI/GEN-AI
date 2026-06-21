from langchain_core.prompts import ChatPromptTemplate,load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)

model=ChatHuggingFace(llm=llm)

# chat_template=ChatPromptTemplate([
#     SystemMessage(content="you are a helpful {domain} expert"),
#     HumanMessage(content="Explain in simpler terms ,what is {topi}")
# ])
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])


prompt=chat_template.invoke({
    'domain':'cricket',
    'topic':'LBW'
})


result=model.invoke(prompt)

print(result.content)