'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.prompts import qa_template, condense_template
from src.llm import build_llm

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

def set_conversation_qa_prompt():
    """
    Prompt template to merge a previous question history with a new question
    """
    prompt = PromptTemplate(input_variables=['chat_history', 'question'], template=condense_template)
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS, # here we say return back the source data
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return dbqa


def build_conversational_retrieval_qa(llm, condense_prompt, final_prompt, vectordb):


    dbqa = ConversationalRetrievalChain.from_llm(
                                        llm = llm,
                                        retriever = vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                        condense_question_prompt = condense_prompt,
                                        # combine_docs_chain_kwargs - used to combine new_standalone_question with the retrieved document, must contain {context}
                                        combine_docs_chain_kwargs = {'prompt': final_prompt},
                                        chain_type = "stuff",
                                        return_source_documents = True,
                                        return_generated_question = True
                                        )
    return dbqa




def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    make_standalone_question_prompt = set_conversation_qa_prompt()
    dbqa = build_conversational_retrieval_qa(llm, make_standalone_question_prompt, qa_prompt, vectordb)

    return dbqa
