from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from flask import Flask, request, jsonify
from waitress import serve
import os

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

def chatbot(input_text):
    
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="indexes")
    
    #load indexes from directory using storage_context 
    query_engne = load_index_from_storage(storage_context).as_query_engine()
    
    response = query_engne.query(input_text)
    
    #returning the response
    return response.response

@app.route('/', methods=['GET'])
def chatbot_start():
    return "Hello from Simplisafe!!"

@app.route('/chatbot/<input_text>', methods=['GET'])
def chatbot_endpoint(input_text):
    response = chatbot(input_text)
    
    return jsonify({'response': response})

# Run the app on Waitress server
if __name__ == '__main__':
    app.run(debug=True, port=8080)
