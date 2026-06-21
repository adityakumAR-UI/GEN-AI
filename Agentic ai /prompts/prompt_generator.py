from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts in a simple and intuitive manner.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

3. Code Examples:
   - If applicable, provide simple code snippets demonstrating the core concept.

4. Real-World Applications:
   - Explain practical use cases and industry applications.

5. Key Contributions:
   - Highlight the major innovations and contributions of the paper.

If certain information is not available in the paper, respond with:
"Insufficient information available."

Ensure the summary is clear, accurate, and aligned with the selected style and length.
""",
input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)

template.save('template.json')