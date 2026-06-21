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

prompt1=PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="summarize the following text {text}",
    input_variables=['text']
)

parser=StrOutputParser()

report_gen_chain=RunnableSequence(prompt1 | model | parser)
branch_chain=RunnableBranch(
    (lambda x :len(x.split())>200,RunnableSequence(prompt2 | model | parser)),
    RunnablePassthrough()
    
)
final_chain=RunnableSequence(report_gen_chain | branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukrain'}))