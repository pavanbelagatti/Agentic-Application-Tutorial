from dotenv import load_dotenv
load_dotenv()

from graph.orchestrator import build_graph

def main():
    topic = input("Enter your startup idea or target market: ")
    print(f"\nRunning Startup Intelligence Agent for: {topic}...\n")
    graph = build_graph(topic)
    result = graph.invoke({})
    print("\n\nGenerated Pitch:\n")
    print(result["pitch"])

if __name__ == "__main__":
    main()