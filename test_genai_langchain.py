from langchain_community.llms import OCIGenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# use default authN method API-key
llm = OCIGenAI(
    model_id="cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaau3zd2dlqcjv4igf6gmfmcxnit62oka5rgg34ryygbnqamnyocta",
)

# response = llm.invoke("Tell me one fact about earth", temperature=0.7)
# print(response)

prompt = PromptTemplate(input_variables=["query"], template="{query}")

llm_chain = LLMChain(llm=llm, prompt=prompt)

response = llm_chain.invoke("what is the capital of france?")
print(response)
