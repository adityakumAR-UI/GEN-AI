from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model = ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template=' give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# prompt= template.format()

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

chain = template | model | parser
result=chain.invoke({
    'topic':'blackhole'
})

print(result)
print(type(result))

# no inforced schema