from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated, Literal, Optional
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model = ChatHuggingFace(llm=llm)
# print("TOKEN =", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


#schema for structured output
# class Review(TypedDict):

#     summary: str
#     sentiment: str

class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

structured_model=model.with_structured_output(Review)

result=structured_model.invoke("""Review the following customer review and extract all information.

Reviewer: Aditya Kumar

I recently purchased the iPhone 16 and overall I am very satisfied with it.
The camera quality is excellent, battery life easily lasts a full day, and the performance is extremely smooth.

However, I found the device to be quite expensive and the charging speed could be better compared to some Android phones.

The design feels premium and the display is bright and vibrant. I would definitely recommend it to anyone looking for a flagship smartphone.

Overall, this was a great purchase and exceeded my expectations.""")

print(result)
print(result['summary'])
print(result['sentiment'])