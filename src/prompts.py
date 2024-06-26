'''
===========================================
        Module: Prompts collection
===========================================
'''
# Note: Precise formatting of spacing and indentation of the prompt template is important for Llama-2-7B-Chat,
# as it is highly sensitive to whitespace changes. For example, it could have problems generating
# a summary from the pieces of context if the spacing is not done correctly

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
# Style #
The response should be clear, concise, and in the form of a straightforward decision.

# Tone # 
Professional and analytical.

# Audience # 
The audience is professional experts in oil and gas industry

Your answer is very important to my career.
Helpful answer:
"""

condense_template = """
    Combine the chat history and follow up question into
    a standalone question.
    You are professional expert in oil and gas industry

    Chat History: {chat_history}
    Follow up question: {question}
    Standalone question:
    """