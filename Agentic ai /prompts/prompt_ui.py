from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=1600
)
model = ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# template = PromptTemplate(
    
#     template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:

# Explanation Style: {style_input}
# Explanation Length: {length_input}

# 1. Mathematical Details:
#    - Include relevant mathematical equations if present in the paper.
#    - Explain the mathematical concepts in a simple and intuitive manner.

# 2. Analogies:
#    - Use relatable analogies to simplify complex ideas.

# 3. Code Examples:
#    - If applicable, provide simple code snippets demonstrating the core concept.

# 4. Real-World Applications:
#    - Explain practical use cases and industry applications.

# 5. Key Contributions:
#    - Highlight the major innovations and contributions of the paper.

# If certain information is not available in the paper, respond with:
# "Insufficient information available."

# Ensure the summary is clear, accurate, and aligned with the selected style and length.
# """,
# input_variables=["paper_input", "style_input", "length_input"],
#     validate_template=True,
# )


template=load_prompt('template.json')




if st.button('Summarize'):
   chain = template | model
   result=chain.invoke({
      'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
      
   })
#    prompt=template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
#     }) 
#    result=model.invoke()
   st.write(result.content)

# user_input= st.text_input("Enter your query about the research paper:")
# if st.button('summarize'):
#     result=model.invoke(user_input)
#     st.write(result.content)