"""
Agent 4: Concept Mapper
Extracts key concepts and their relationships from a paper
to build an interactive knowledge graph.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ConceptNode(BaseModel):
    """A node in the concept knowledge graph."""
    id: str = Field(description="Unique short identifier (lowercase, underscores, e.g. 'self_attention')")
    label: str = Field(description="Human-readable display name (e.g. 'Self-Attention')")
    category: str = Field(description="One of: method, dataset, metric, concept, result, problem")
    importance: int = Field(description="1-5, how central to the paper (5=core contribution)", ge=1, le=5)


class ConceptEdge(BaseModel):
    """An edge connecting two concepts in the knowledge graph."""
    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    relationship: str = Field(
        description="Relationship type, e.g.: 'uses', 'improves', 'evaluates_on', 'produces', 'solves', 'extends', 'compared_to', 'part_of', 'enables'"
    )


class ConceptMap(BaseModel):
    """Complete concept map with nodes and edges."""
    nodes: list[ConceptNode] = Field(description="10-20 key concept nodes")
    edges: list[ConceptEdge] = Field(description="Meaningful edges between concepts")


class ConceptMapperAgent:
    """Agent 4: Extracts concepts and relationships for knowledge graph visualization."""

    def __init__(self):
        self.llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

    def map_concepts(self, parsed_paper: dict) -> dict:
        """
        Extract concept map from a parsed paper.
        
        Args:
            parsed_paper: Output from PaperParserAgent.parse()
            
        Returns:
            dict with 'nodes' and 'edges' for the knowledge graph
        """
        logger.info(f"Agent 4 (Concept Mapper): Mapping '{parsed_paper['parsed']['title']}'")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledge graph specialist. Extract the key concepts 
            from this research paper and their relationships.
            
            Create a concept map with:
            - 10-15 nodes representing key concepts, methods, datasets, and metrics
            - Meaningful directed edges showing relationships between concepts
            - Proper categorization: method, dataset, metric, concept, result, problem
            - Importance ratings: 5=core contribution, 4=important, 3=supporting, 2=context, 1=peripheral
            
            Rules:
            - Every node MUST connect to at least one other node
            - Use lowercase_underscore format for IDs (e.g., 'self_attention', 'bleu_score')
            - Edge source and target must reference valid node IDs
            - Focus on the MOST important concepts — don't list everything
            - Include the paper's main problem, proposed method, datasets used, and key results"""),
            ("user", """Paper: {title}
            
            Abstract: {abstract}
            
            Problem: {problem}
            
            Methodology: {methodology}
            
            Key Terms: {key_terms}
            
            Key Results: {results}
            
            Generate an accurate concept map.""")
        ])

        structured_llm = self.llm.with_structured_output(ConceptMap)

        result = structured_llm.invoke(
            prompt.format_messages(
                title=parsed_paper["parsed"]["title"],
                abstract=parsed_paper["metadata"]["abstract"],
                problem=parsed_paper["parsed"]["problem_statement"],
                methodology=parsed_paper["parsed"]["methodology_summary"],
                key_terms=", ".join(parsed_paper["parsed"]["key_terms"]),
                results="\n".join(parsed_paper["parsed"]["key_results"]),
            )
        )

        # Validate edges reference valid nodes
        node_ids = {n.id for n in result.nodes}
        valid_edges = [
            e for e in result.edges
            if e.source in node_ids and e.target in node_ids
        ]

        if len(valid_edges) < len(result.edges):
            logger.warning(
                f"  Removed {len(result.edges) - len(valid_edges)} edges with invalid node references"
            )

        concept_map = {
            "nodes": [n.model_dump() for n in result.nodes],
            "edges": [e.model_dump() for e in valid_edges],
        }

        logger.info(f"  ✅ Mapped {len(concept_map['nodes'])} concepts, {len(concept_map['edges'])} relationships")
        return concept_map
