from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import logging
import sys
import os
from llama_index.llms import Ollama

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = Ollama(model="llama2", request_timeout=30.0)

service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

# print(os.environ.get("OPENAI_API_BASE"))
# print(os.environ.get("OPENAI_API_KEY"))
