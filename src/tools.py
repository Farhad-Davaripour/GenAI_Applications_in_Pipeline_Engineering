from llama_index.core.query_engine import CustomQueryEngine
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import TextNode, ImageNode, NodeWithScore, MetadataMode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)

# import importlib
# from src import prompt_temp
# importlib.reload(prompt_temp)
from src.prompt_temp import qa_system_prompt

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

from llama_index.llms.openai import OpenAI as llma_OpenAI
llm = llma_OpenAI(model="gpt-4o")

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
    