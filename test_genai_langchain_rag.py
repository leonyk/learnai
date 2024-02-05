from langchain_community.llms import OCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-light-v2.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaau3zd2dlqcjv4igf6gmfmcxnit62oka5rgg34ryygbnqamnyocta",
)

vectorstore = FAISS.from_texts(
    [
        "Larry Ellison co-founded Oracle Corporation in 1977 with Bob Miner and Ed Oates.",
        "Oracle Corporation is an American multinational computer technology company headquartered in Austin, Texas, United States.",
    ],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}
 
Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# use default authN method API-key
llm = OCIGenAI(
    model_id="cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaau3zd2dlqcjv4igf6gmfmcxnit62oka5rgg34ryygbnqamnyocta",
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("when was oracle founded?"))
print(chain.invoke("where is oracle headquartered?"))
