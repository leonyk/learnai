
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("state_of_the_union.txt")
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
llm = ChatOpenAI(model="gpt-3.5-turbo")

questions = [
    "Who is the speaker",
    "What did the president say about Ketanji Brown Jackson",
    "What are the threats to America",
    "Who are mentioned in the speech",
    "Who is the vice president",
    "How many projects were announced",
]

for query in questions:
    print("Query:", query)
    print("Answer:", index.query(query, llm=llm))
