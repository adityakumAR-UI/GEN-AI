from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableSequence

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model = ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt1=PromptTemplate(
    template="explain the following joke -{text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain=RunnableSequence(prompt | model | parser | prompt1 | model | parser)

print(chain.invoke({'topic':'ai'}))


