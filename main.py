import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='What is the flange faces you know?',
                        help='Enter the question to pass into the LLM')
    args = parser.parse_args()

    # Setup DBQA
    start = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({'question': args.input, 'chat_history': "" })
    end = timeit.default_timer()

    print(f'\nAnswer: {response["answer"]}')
    print('='*50)

    # Process source documents
    if cfg.RETURN_SOURCE_DOCUMENTS:
        source_docs = response['source_documents']
        for i, doc in enumerate(source_docs):
            print(f'\nSource Document {i+1}\n')
            print(f'Source Text: {doc.page_content}')
            print(f'Document Name: {doc.metadata["source"]}')
            print(f'Page Number: {doc.metadata["page"]}\n')
            print('='* 60)

    if cfg.RETURN_GENERATED_QUESTION:
        print(f'Generated question {response["generated_question"]} ')
        print('=' * 60)

    print(f"Time to retrieve response: {end - start}")
