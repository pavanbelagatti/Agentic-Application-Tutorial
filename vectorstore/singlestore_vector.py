from langchain_singlestore.vectorstores import SingleStoreVectorStore
from langchain_openai import OpenAIEmbeddings
import os

def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    return SingleStoreVectorStore(
        embedding=embeddings,
        host=os.environ["SINGLESTORE_HOST"]
    )