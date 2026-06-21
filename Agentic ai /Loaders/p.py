from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableSequence,RunnablePassthrough

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model = ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='write a detailed score summarization about {person}',
    input_variables=['person']
)
parser=StrOutputParser()

chain=RunnableSequence(prompt | model | parser)
print(chain.invoke({'person':"rohit sharma"}))