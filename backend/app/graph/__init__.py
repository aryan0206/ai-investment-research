from langgraph.graph import StateGraph, END
from app.graph.state import ResearchState
from app.graph.nodes import (
    planner_node,
    retrieval_node,
    synthesizer_node,
    report_generator_node,
)


def build_research_graph():
    """
    Phase 1: Linear graph.
    Planner → Retrieval → Synthesizer → Report Generator

    Phase 2 will replace this with parallel research nodes
    (news, fundamentals, filings, signals) feeding into the synthesizer.
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("report_generator", report_generator_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retrieval")
    workflow.add_edge("retrieval", "synthesizer")
    workflow.add_edge("synthesizer", "report_generator")
    workflow.add_edge("report_generator", END)

    return workflow.compile()


# Compiled once at startup — reused across all requests
research_graph = build_research_graph()
