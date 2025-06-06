from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from typing import Literal

class State(TypedDict):
    user_message : str
    ai_message : str
    is_coding_question: bool
    
def detect_query(state: State):
    user_message = state.get("user_message")
    
    # OpenAI Call
    state.is_coding_question = True
    
    return state

def route_edge(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    is_coding_question = state.get("is_coding_question")

    if is_coding_question:
        return "solve_coding_question"
    else:
        return "solve_simple_question"
    
def solve_coding_question(state : State):
    user_message = state.get("user_message")
    
    # OpenAI Call (Coding question gpt-4.o)
    state.ai_message = "Here is your coding question answer."
    
    return state

def solve_simple_question(state : State):
    user_message = state.get("user_message")
    
    # OpenAI Call (Coding question gpt-mini)
    state.ai_message = "Please ask some coding related question"
    
    return state
    
    
graph_builder = StateGraph(State)

graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_edge("route_edge", route_edge)


graph_builder.add_edge(START, "detect_query")
graph_builder.add_edge("detect_query", "route_edge")
graph_builder.add_conditional_edge("route_edge", "solve_coding_question")
graph_builder.add_conditional_edge("route_edge", "solve_simple_question")
