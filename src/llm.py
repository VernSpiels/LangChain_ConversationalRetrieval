'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
#from langchain.llms import CTransformers
# working with langchain_community https://python.langchain.com/docs/integrations/llms/llamacpp/
from langchain_community.llms import LlamaCpp
from dotenv import find_dotenv, load_dotenv
import box
import yaml

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_llm():
    #local LLama_cpp
    llm = LlamaCpp(model_path = cfg.MODEL_BIN_PATH,
                   temperature = cfg.TEMPERATURE,
                   max_tokens=512,
                   f16_kv=True,
                   n_batch=256,
                   n_parts=10,
                   n_threads=10,
                   n_ctx=512,
                   top_p=cfg.TOP_P,
                   repeat_penalty =1.1,
                   last_n_tokens_size = 64,
                   streaming=True
                   )



    return llm

print('Done')
