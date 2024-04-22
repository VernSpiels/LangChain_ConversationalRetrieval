import box
import timeit
import yaml
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
from glob import glob

from PIL import Image

import streamlit as st
from streamlit_chat import message


# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Function for conversational chat
def conversational_chat(dbqa_chain,query):
    response = dbqa_chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, response["answer"]))
    return response


# Making web UI
def make_web_ui():
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize generation_time history
    if 'generated_time' not in st.session_state:
        st.session_state['generated_time'] = ['0 sec']

    # Initialize generation_time history
        if 'summarazied_question' not in st.session_state:
            st.session_state['summarazied_question'] = [' ']

    # Initialize generation_time history
        if 'source_documents' not in st.session_state:
            st.session_state['source_documents'] = [' ']

    # Initialize messages
    if 'generated' not in st.session_state:
        try:
            files_in_db = ' '.join([i.replace('data\\', '') for i in glob(cfg.DATA_PATH+'/*.pdf')]) #' '.join([i.replace(cfg.DATA_PATH+'\\', '/') for i in glob(cfg.DATA_PATH+'*.pdf')])
        except:
            files_in_db = 'files in data path'

        st.session_state['generated'] = ["Hello ! Ask me llama_2 about " + files_in_db + "   🤗"]
        #st.session_state['generated'] = ["Hello ! Ask me llama_2 about files in data path 🤗"]

    # Initialize chat_bot_responses
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]



if __name__ == "__main__":

    # Set the title for the Streamlit app
    st.title("Llama2 Chat with PDF - 🦙")

    make_web_ui()

    # Setup DBQA
    dbqa_chain = setup_dbqa()

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to PDF data 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            start = timeit.default_timer()
            output = conversational_chat(dbqa_chain, user_input)
            end = timeit.default_timer()

            st.session_state['summarazied_question'].append(f'I have summarized your question: {output["generated_question"]} ')
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output["answer"])
            st.session_state['generated_time'].append(f"Time to retrieve response: {round(end - start,2)} sec.")

            # adding source_documents
            doc = output['source_documents'][0]
            st.session_state['source_documents'].append(
                f'Source Text: {doc.page_content}\nDocument Name: {doc.metadata["source"]}\nPage Number: {doc.metadata["page"]}')

    bot_image = Image.open("llama.jpg")

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="shapes")
                if i>0 : message(st.session_state["summarazied_question"][i], key=str(i) + '_ai_1', avatar_style="🧑‍💻")
                message(st.session_state["generated"][i], key=str(i)+ '_ai', avatar_style="fun-emoji",seed=7)
                if i>0 :
                    message(st.session_state["source_documents"][i], key=str(i) + '_ai_2', avatar_style="avataaars")
                    message(st.session_state["generated_time"][i], key=str(i)+ '_assistant', avatar_style="adventurer-neutral")









