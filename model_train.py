from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

def construct_index(directory_path):
    # set number of output tokens
    num_outputs = 256

    _llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=_llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    
    #Directory in which the indexes will be stored
    index.storage_context.persist(persist_dir="indexes")

    return index

#Constructing indexes based on the documents in data folder
#This can be skipped if you have already trained your app and need to re-run it
construct_index("training_data")