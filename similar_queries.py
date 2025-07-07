from dotenv import load_dotenv
load_dotenv()

from vectorstore.singlestore_vector import get_vectorstore

def search_similar_queries(query: str):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n🔍 Top 3 matches for: '{query}'\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:250]}...\n")

if __name__ == "__main__":
    user_query = input("Enter a topic to search similar past analyses: ")
    search_similar_queries(user_query)