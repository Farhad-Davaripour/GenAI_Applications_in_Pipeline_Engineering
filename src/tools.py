import nest_asyncio
nest_asyncio.apply()
from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex


def get_nodes(docs):
    """Split docs into nodes, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split("\n---\n")
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)

    return nodes

def calculate_circumferential_stress(P, D, t_n):
    """
    Calculate the circumferential (hoop) stress in a pipeline based on CSA Z662 standard (Section 4.8.3).
    
    Args:
    P (float): Design pressure of the pipeline (in MPa)
    D (float): Outside diameter of the pipe (in mm)
    t_n (float): Pipe nominal wall thickness, less allowances (in mm)
    
    Returns:
    float: Circumferential (hoop) stress (in MPa)
    """
    S_h = (P * D) / (2 * t_n)
    return S_h
