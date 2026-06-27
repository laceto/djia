"""Playlist generator for creating optimal DJ sets."""

from typing import Dict, List, Optional
from heapq import heappush, heappop
from .transition_mapper import build_transition_graph, score_transition


def generate_playlist(
    all_tracks: Dict[int, Dict],
    start_track_id: int,
    end_track_id: int,
    num_steps: int = 5,
) -> Optional[List[int]]:
    """
    Generate an optimal transition path between two tracks.

    Uses a simple greedy algorithm to find the best intermediate tracks
    that transition well from start to end.

    Args:
        all_tracks: Dictionary mapping track_id -> track_features_dict
        start_track_id: ID of starting track
        end_track_id: ID of ending track
        num_steps: Number of intermediate steps (including start/end)

    Returns:
        List of track IDs representing the optimal path, or None if no path found
        Example: [5, 12, 8, 3, 9] for a 5-track path

    Example:
        >>> playlist = generate_playlist(tracks, start_id=1, end_id=10, num_steps=5)
        >>> # Bridge from track 1 to track 10 in 5 steps
    """
    if start_track_id not in all_tracks or end_track_id not in all_tracks:
        return None

    if num_steps < 2:
        return [start_track_id] if start_track_id == end_track_id else None

    # Build transition graph
    graph = build_transition_graph(all_tracks)

    # Use Dijkstra-based approach to find best path
    playlist = _dijkstra_path(
        all_tracks, graph, start_track_id, end_track_id, num_steps
    )

    return playlist


def _dijkstra_path(
    all_tracks: Dict[int, Dict],
    graph: Dict[int, List[tuple]],
    start_id: int,
    end_id: int,
    num_steps: int,
) -> Optional[List[int]]:
    """
    Find optimal path using modified Dijkstra's algorithm.

    Optimizes for:
    1. Transition quality (high scores)
    2. Path length (exactly num_steps)
    """
    # Priority queue: (negative_score, current_id, path_length, path)
    heap = [(-1.0, start_id, 1, [start_id])]
    visited = {}  # (node, path_length) -> best_score

    best_path = None
    best_score = float('-inf')

    while heap:
        neg_score, current_id, path_length, path = heappop(heap)
        score = -neg_score

        # Reached end step
        if path_length == num_steps:
            if current_id == end_id:
                if score > best_score:
                    best_score = score
                    best_path = path
            continue

        # Pruning: skip if we've seen this state with better score
        state = (current_id, path_length)
        if state in visited and visited[state] >= score:
            continue
        visited[state] = score

        # Explore neighbors from graph
        if current_id in graph:
            for next_id, transition_score in graph[current_id][:10]:  # Top 10 transitions
                if next_id in path:  # Avoid loops
                    continue

                # Boost score if moving toward end_id
                distance_to_end = _simple_distance(
                    all_tracks[next_id], all_tracks[end_id]
                )
                next_score = score + transition_score - (distance_to_end * 0.1)

                new_path = path + [next_id]
                heappush(heap, (-next_score, next_id, path_length + 1, new_path))

    # If we didn't reach end_id exactly, find closest path
    if best_path is None:
        # Fallback: greedy approach
        best_path = _greedy_path(all_tracks, graph, start_id, end_id, num_steps)

    return best_path


def _simple_distance(track_a: Dict, track_b: Dict) -> float:
    """Simple distance metric between two tracks (0-1)."""
    # BPM distance
    bpm_a = track_a.get('tempo', 120)
    bpm_b = track_b.get('tempo', 120)
    bpm_dist = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b, 1)

    # Mood distance (if available)
    mood_a = track_a.get('mood', {})
    mood_b = track_b.get('mood', {})

    if mood_a and mood_b:
        common_moods = set(mood_a.keys()) & set(mood_b.keys())
        if common_moods:
            mood_dist = sum(
                abs(mood_a.get(m, 0) - mood_b.get(m, 0))
                for m in common_moods
            ) / len(common_moods)
        else:
            mood_dist = 0.5
    else:
        mood_dist = 0.0

    return (bpm_dist + mood_dist) / 2


def _greedy_path(
    all_tracks: Dict[int, Dict],
    graph: Dict[int, List[tuple]],
    start_id: int,
    end_id: int,
    num_steps: int,
) -> List[int]:
    """
    Greedy fallback: always pick the best next track.

    Doesn't guarantee optimal path but is fast and simple.
    """
    path = [start_id]
    current_id = start_id
    used = {start_id}

    for _ in range(num_steps - 2):  # -2 for start and end
        if current_id not in graph or not graph[current_id]:
            # Dead end, restart from best option
            available = [
                (tid, tid == end_id)
                for tid, _ in graph.get(current_id, [])
                if tid not in used
            ]
            if not available:
                break
            next_id = available[0][0]
        else:
            # Pick best transition that's not already used
            next_id = None
            for candidate_id, _ in graph[current_id]:
                if candidate_id not in used:
                    next_id = candidate_id
                    break

            if next_id is None:
                break

        path.append(next_id)
        used.add(next_id)
        current_id = next_id

    # Add end track if not already there
    if end_id not in used:
        path.append(end_id)

    # Pad or trim to exact num_steps
    if len(path) < num_steps:
        # Pad with intermediate tracks
        while len(path) < num_steps and current_id in graph:
            for next_id, _ in graph[current_id]:
                if next_id not in used:
                    path.insert(-1, next_id)  # Insert before end
                    used.add(next_id)
                    current_id = next_id
                    break
            if len(path) >= num_steps:
                break

    # Ensure we have exactly num_steps and end with end_track
    path = path[:num_steps - 1] + [end_id]

    return path[:num_steps]


def playlist_summary(
    playlist: List[int],
    all_tracks: Dict[int, Dict],
) -> Dict:
    """
    Generate a summary of playlist characteristics.

    Args:
        playlist: List of track IDs
        all_tracks: All track features

    Returns:
        Dict with summary statistics
    """
    if not playlist:
        return {}

    tracks = [all_tracks[tid] for tid in playlist if tid in all_tracks]

    # BPM arc
    bpms = [t.get('tempo', 120) for t in tracks]
    start_bpm = bpms[0] if bpms else 120
    end_bpm = bpms[-1] if bpms else 120

    # Transition scores
    transition_scores = []
    for i in range(len(playlist) - 1):
        if playlist[i] in all_tracks and playlist[i + 1] in all_tracks:
            score = score_transition(
                all_tracks[playlist[i]],
                all_tracks[playlist[i + 1]],
            )
            transition_scores.append(score.overall_score)

    avg_transition = (
        sum(transition_scores) / len(transition_scores)
        if transition_scores
        else 0.0
    )

    return {
        'length': len(playlist),
        'start_bpm': round(start_bpm, 1),
        'end_bpm': round(end_bpm, 1),
        'bpm_arc': end_bpm - start_bpm,
        'avg_transition_score': round(avg_transition, 3),
        'min_transition_score': round(min(transition_scores), 3) if transition_scores else 0,
        'max_transition_score': round(max(transition_scores), 3) if transition_scores else 0,
    }
