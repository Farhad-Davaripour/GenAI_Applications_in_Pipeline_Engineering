# Import standard libraries
import math
# import importlib

# Llama Index core dependencies
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import TextNode, ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

# Llama Index multi-modal LLM and embedding dependencies
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.embeddings.openai import OpenAIEmbedding

# Reload and import custom modules
from src import prompt_temp
# importlib.reload(prompt_temp)
from src.prompt_temp import qa_system_prompt

# Llama Index OpenAI LLM initialization
from llama_index.llms.openai import OpenAI as llma_OpenAI
llm = llma_OpenAI(model="gpt-4o")

# Initialize the OpenAI embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

Settings.llm = llm
Settings.embed_model = embed_model

# build storage context
storage_context = StorageContext.from_defaults(persist_dir="storage_nodes")
# load index
index = load_index_from_storage(storage_context, index_id="vector_index")

class MARKDOWN_QUERY_ENGINE(CustomQueryEngine):

    qa_prompt: PromptTemplate

    def __init__(self, qa_prompt = qa_system_prompt(), **kwargs) -> None:
        super().__init__(qa_prompt=qa_prompt, **kwargs)

    def custom_query(self, query_str: str):
        retriever = index.as_retriever(similarity_top_k=5)
        # retrieve text nodes
        text_nodes = retriever.retrieve(query_str)

        context_str = "\n\n".join(
                    [r.get_content(metadata_mode=MetadataMode.LLM) for r in text_nodes]
                )
        fmt_prompt = self.qa_prompt.format(context_str=context_str, query_str=query_str)
        llm_response = llm.complete(prompt=fmt_prompt)
        return Response(response=str(llm_response), source_nodes=text_nodes)

def calculate_axial_soil_force(D, c, H, gamma, alpha, K_0, delta):
    """
    Calculate the maximum axial soil force per unit length of a buried pipeline (T_u)
    based on the American Lifelines Alliance (ALA) 2005 guidelines.

    IMPORTANT: 
    This function is solely used to calculate the maximum axial soil force per unit length of a pipe 
    (T_u) as per the given equation. To understand the derivation, detailed components of the formula, 
    and specific conditions or limitations, you should refer to the relevant sections in the ALA guidelines.
    This tool does not replace the guidelines but serves as a calculator for this specific equation.
    Also, make sure the units that are input into the function are consistent with the units mentioned under the Args section below:

    Args:
    D (float): Pipe outside diameter (in meters)
    c (float): Soil cohesion representative of the soil backfill (in kPa)
    H (float): Depth to pipe centerline (in meters)
    gamma (float): Effective unit weight of soil (in kN/m^3)
    alpha (float): Adhesion factor
    K_0 (float): Coefficient of earth pressure at rest
    delta (float): Interface friction angle between the pipe and the soil (in radians)

    Returns:
    float: Maximum axial soil force per unit length of pipe (T_u) in kN/m
    """
    # Calculate the maximum axial soil force per unit length (T_u)
    T_u = math.pi * D * alpha * c + math.pi * D * H * gamma * ((1 + K_0) / 2) * math.tan(delta * math.pi / 180)

    return T_u
