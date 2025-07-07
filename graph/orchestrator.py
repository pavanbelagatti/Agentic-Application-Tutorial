from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.research_agent import get_research_agent
from agents.summarizer_agent import summarize_documents
from agents.vector_agent import store_documents
from agents.pitch_generator_agent import generate_pitch
from langchain_core.documents import Document

# Define the graph state schema
class GraphState(TypedDict):
    docs: list[Document]
    summary: str
    pitch: str

# Build LangGraph workflow
def build_graph(user_topic: str):
    builder = StateGraph(GraphState)

    # 🟦 Node 1: Research using Tavily
    def research_node(state: GraphState) -> GraphState:
        agent = get_research_agent()
        result = agent.invoke(f"Research latest trends and competitors for {user_topic}")

        # ✅ Fix: extract output string from the result dict
        output_text = result.get("output", "") if isinstance(result, dict) else str(result)

        docs = [Document(page_content=output_text)]
        store_documents(docs)
        return {"docs": docs}

    # 🟩 Node 2: Summarize documents
    def summarize_node(state: GraphState) -> GraphState:
        return {"summary": summarize_documents(state["docs"])}

    # 🟨 Node 3: Generate pitch
    def pitch_node(state: GraphState) -> GraphState:
        return {"pitch": generate_pitch(state["summary"])}

    # 🧠 Build the LangGraph DAG
    builder.add_node("research", research_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("generate_pitch_node", pitch_node)

    builder.set_entry_point("research")
    builder.add_edge("research", "summarize")
    builder.add_edge("summarize", "generate_pitch_node")
    builder.add_edge("generate_pitch_node", END)

    return builder.compile()