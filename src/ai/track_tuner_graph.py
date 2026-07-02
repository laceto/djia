"""LangGraph assembly for Track Tuner agent."""

from typing import Literal
from langgraph.graph import END, START, StateGraph
from .track_tuner_state import TrackTunerState, BatchTrackTunerState
from .track_tuner_nodes import (
    load_track,
    initialize_config,
    analyze_track,
    evaluate_quality,
    suggest_tuning,
    finalize,
)


def build_single_track_graph():
    """Build graph for analyzing and tuning a single track."""
    g = StateGraph(TrackTunerState)

    # Add nodes
    g.add_node("load_track", load_track)
    g.add_node("initialize_config", initialize_config)
    g.add_node("analyze_track", analyze_track)
    g.add_node("evaluate_quality", evaluate_quality)
    g.add_node("suggest_tuning", suggest_tuning)
    g.add_node("finalize", finalize)

    # Define routing logic
    def should_iterate(state: TrackTunerState) -> Literal["analyze_track", "finalize"]:
        """Decide whether to tune and re-analyze or finalize."""
        satisfied = state.get("satisfied", False)
        iterations = state.get("iterations_completed", 0)
        max_iterations = state.get("max_iterations", 3)

        if satisfied or iterations >= max_iterations:
            return "finalize"
        else:
            return "analyze_track"

    # Build edges
    g.add_edge(START, "load_track")
    g.add_edge("load_track", "initialize_config")
    g.add_edge("initialize_config", "analyze_track")
    g.add_edge("analyze_track", "evaluate_quality")
    g.add_conditional_edges(
        "evaluate_quality",
        should_iterate,
        {
            "analyze_track": "suggest_tuning",
            "finalize": "finalize",
        },
    )
    g.add_edge("suggest_tuning", "analyze_track")
    g.add_edge("finalize", END)

    return g.compile()


def build_batch_graph():
    """Build graph for analyzing multiple tracks."""
    g = StateGraph(BatchTrackTunerState)

    # Add nodes
    g.add_node("process_track", process_single_track)
    g.add_node("aggregate_results", aggregate_results)

    # Build edges
    g.add_edge(START, "process_track")
    g.add_conditional_edges(
        "process_track",
        lambda state: "process_track" if state["current_track_index"] < len(state["track_paths"]) else "aggregate_results",
    )
    g.add_edge("aggregate_results", END)

    return g.compile()


def process_single_track(state: BatchTrackTunerState, config) -> dict:
    """Process a single track in batch mode."""
    from langchain_core.messages import AIMessage

    idx = state["current_track_index"]
    tracks = state["track_paths"]

    if idx >= len(tracks):
        return {"current_track_index": idx}

    track_path = tracks[idx]

    # Create single-track state and run
    single_state: TrackTunerState = {
        "track_path": track_path,
        "track_name": None,
        "bpm": None,
        "duration": None,
        "current_segments": None,
        "current_quality": None,
        "current_config": None,
        "initial_config": None,
        "iterations_completed": 0,
        "max_iterations": 3,
        "satisfied": False,
        "reason": None,
        "analysis_history": [],
        "recommendations": [],
        "messages": [],
    }

    single_config = {"configurable": {"preset": state["config_preset"]}}

    # Run single track graph
    graph = build_single_track_graph()
    result = graph.invoke(single_state, config=single_config)

    # Convert result to TrackAnalysisResult
    from .track_tuner_state import TrackAnalysisResult

    analysis_result: TrackAnalysisResult = {
        "track_path": track_path,
        "track_name": result.get("track_name", "unknown"),
        "bpm": result.get("bpm", 0),
        "duration": result.get("duration", 0),
        "num_segments": result.get("current_quality", {}).get("num_segments", 0),
        "avg_bars": result.get("current_quality", {}).get("avg_bars_per_segment", 0),
        "config": result.get("current_config", {}),
        "quality_score": result.get("current_quality", {}).get("quality_score", 0),
        "satisfied": result.get("satisfied", False),
        "iterations_used": result.get("iterations_completed", 0),
    }

    msg = f"[ProcessTrack] Finished track {idx + 1}/{len(tracks)}: {analysis_result['track_name']}"

    return {
        "current_track_index": idx + 1,
        "results": [analysis_result],
        "messages": [AIMessage(content=msg)],
    }


def aggregate_results(state: BatchTrackTunerState, config) -> dict:
    """Aggregate results from all tracks."""
    from langchain_core.messages import AIMessage

    results = state["results"]
    satisfied_count = sum(1 for r in results if r["satisfied"])

    msg = f"[AggregateResults] Processed {len(results)} tracks | {satisfied_count} satisfied"

    return {
        "messages": [AIMessage(content=msg)],
    }


def run_single_track(track_path: str, preset: str = "minimal", max_iterations: int = 3):
    """Run single track tuning."""
    graph = build_single_track_graph()

    initial_state: TrackTunerState = {
        "track_path": track_path,
        "track_name": None,
        "bpm": None,
        "duration": None,
        "current_segments": None,
        "current_quality": None,
        "current_config": None,
        "initial_config": None,
        "iterations_completed": 0,
        "max_iterations": max_iterations,
        "satisfied": False,
        "reason": None,
        "analysis_history": [],
        "recommendations": [],
        "messages": [],
    }

    config = {"configurable": {"preset": preset}}

    return graph.invoke(initial_state, config=config)


def run_batch_tracks(track_paths: list, preset: str = "minimal"):
    """Run batch tuning over multiple tracks."""
    graph = build_batch_graph()

    initial_state: BatchTrackTunerState = {
        "track_paths": track_paths,
        "current_track_index": 0,
        "results": [],
        "config_preset": preset,
        "messages": [],
    }

    config = {}

    return graph.invoke(initial_state, config=config)
