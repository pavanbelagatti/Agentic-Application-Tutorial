from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

def summarize_documents(docs):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(docs)