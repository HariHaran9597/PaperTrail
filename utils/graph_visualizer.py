"""
Knowledge Graph Visualizer
Builds interactive Pyvis knowledge graphs from concept maps.
"""

from pyvis.network import Network
import os
import logging

logger = logging.getLogger(__name__)

# Color scheme per concept category
CATEGORY_COLORS = {
    "method": "#6366f1",     # Indigo
    "dataset": "#f59e0b",    # Amber
    "metric": "#10b981",     # Emerald
    "concept": "#3b82f6",    # Blue
    "result": "#ef4444",     # Red
    "problem": "#8b5cf6",    # Violet
}

CATEGORY_SHAPES = {
    "method": "diamond",
    "dataset": "database",
    "metric": "triangle",
    "concept": "dot",
    "result": "star",
    "problem": "square",
}


def build_knowledge_graph(concept_map: dict, output_path: str = "concept_graph.html") -> str:
    """
    Build an interactive knowledge graph visualization using Pyvis.
    
    Args:
        concept_map: dict with 'nodes' and 'edges' from ConceptMapperAgent
        output_path: Path to save the HTML file
        
    Returns:
        Path to the generated HTML file
    """
    logger.info(f"Building knowledge graph with {len(concept_map['nodes'])} nodes, {len(concept_map['edges'])} edges")

    net = Network(
        height="550px",
        width="100%",
        bgcolor="#0f172a",    # Dark slate background
        font_color="#e2e8f0",  # Light text
        directed=True,
    )

    # Physics settings for nice layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09,
    )

    # Add nodes
    for node in concept_map["nodes"]:
        color = CATEGORY_COLORS.get(node["category"], "#94a3b8")
        size = 15 + (node["importance"] * 8)  # 23 to 55

        net.add_node(
            node["id"],
            label=node["label"],
            color=color,
            size=size,
            shape=CATEGORY_SHAPES.get(node["category"], "dot"),
            title=f"<b>{node['label']}</b><br>Category: {node['category']}<br>Importance: {'⭐' * node['importance']}",
            font={"size": 12 + node["importance"] * 2, "color": "#e2e8f0"},
            borderWidth=2,
            borderWidthSelected=4,
        )

    # Add edges
    for edge in concept_map["edges"]:
        net.add_edge(
            edge["source"],
            edge["target"],
            label=edge["relationship"],
            color="#475569",
            arrows="to",
            font={"size": 10, "color": "#94a3b8", "align": "middle"},
            width=2,
            smooth={"type": "curvedCW", "roundness": 0.2},
        )

    # Save the graph
    net.save_graph(output_path)
    logger.info(f"  ✅ Knowledge graph saved to {output_path}")

    return output_path
