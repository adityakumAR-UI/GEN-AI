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

llm1= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model1 = ChatHuggingFace(llm=llm1)

def word_count(text):
    return len(text.split())

# runnable_word=RunnableLambda(word_count)

# print(runnable_word.invoke('hi deepa , its aditya here'))

prompt1=PromptTemplate(
    template='write a joke about{topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt1 | model1 | parser)

# parallel_chain=RunnableParallel({
#     'joke':RunnablePassthrough(),
#     'word_count':RunnableLambda(word_count)

# })

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x : len(x.split()))

})
final_chain=RunnableSequence(joke_gen_chain | parallel_chain)
result=final_chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])



print(final_result)


