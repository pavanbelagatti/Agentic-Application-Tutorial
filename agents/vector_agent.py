from vectorstore.singlestore_vector import get_vectorstore

def store_documents(docs):
    vs = get_vectorstore()
    vs.add_documents(docs)