from llama_index.llms import Ollama

llm = Ollama(model="llama2", request_timeout=30.0)

# resp = llm.complete("Who is Paul Graham?")
# print(resp)

resp = llm.stream_complete("Who is Paul Graham?")
for r in resp:
    print(r.delta, end="")
