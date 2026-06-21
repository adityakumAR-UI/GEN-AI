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

llm1= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=512
)

model2=ChatHuggingFace(llm=llm2)


prompt1=PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1 | model1 | parser),
    'linkedin': RunnableSequence(prompt2 | model1 | parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])