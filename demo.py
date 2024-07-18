import os
import streamlit as st
import pickle
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import AgentRunner, ReActAgentWorker, ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import importlib
from src import tools
from src import prompt_temp
importlib.reload(tools)
importlib.reload(prompt_temp)
from src.tools import get_nodes, calculate_circumferential_stress
from src.prompt_temp import react_system_prmopt
from dotenv import load_dotenv
load_dotenv(override=True)

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the API keys
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the embeddings and generative models
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4o")

Settings.embed_model = embed_model
Settings.llm = llm

# Load the binary version of section 4 of CSA Z662
with open("docs/design_section.pkl", "rb") as f:
    loaded_documents_gpt4o = pickle.load(f)

# Convert the documents into nodes
nodes = get_nodes(loaded_documents_gpt4o)

# Store the nodes in a VectorStoreIndex
CSA_pipe_design_standard_index = VectorStoreIndex(nodes)

# Create a query engine from the index
query_engine = CSA_pipe_design_standard_index.as_query_engine(
    streaming=True, similarity_top_k=6
)

# Create the RAG tool which retrieves information from the CSA Z662 pipe design standard
CSA_pipe_design_standard = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="CSA_pipe_design_standard",
        description=(
            "This tool is essential for retrieving specific context from the CSA Z662 pipe design section. It is crucial for answering detailed user queries with accurate and specific information from the indexed data. Always use this tool when the query pertains to specifics within the CSA Z662 pipe design section. Do NOT select if the question asks for a general summary of the data."
        ),
    ),
)

# Create the circumferential stress calculator tool
circumferential_stress_tool = FunctionTool.from_defaults(
    fn=calculate_circumferential_stress,
    name="circumferential_stress_calculator",
    description="Calculates the circumferential (hoop) stress in a pipeline based on CSA Z662 standard, given the design pressure, outside diameter, and pipe nominal wall thickness less allowances.",
)

# Create the ReAct agent and add the tools
agent = ReActAgent.from_tools(
    [CSA_pipe_design_standard, circumferential_stress_tool], llm=llm, verbose=True
)

# =============================================================================
# Streamlit App
# =============================================================================

# Set the title of the app
st.title("CSA-Z662 Pipe Design Standard Assistant")

# Set the default query
default_query = os.getenv("DEFAULT_QUERY")

# Create a text area for the user to input their query
query = st.text_area("Enter your query here:", value=default_query)

# Load a custom system prompt
react_system_prompt = react_system_prmopt()

# Update the agent with the custom system prompt
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

# Reset the agent
agent.reset()

# Create a task
task = agent.create_task(query)

# Iterate over the thought, action, and observation steps to complete the task
if st.button("Submit"):
    with st.spinner("Processing your query..."):
        step_output = agent.run_step(task.task_id)

        # Check whether the task is complete
        while step_output.is_last == False:
            step_output = agent.run_step(task.task_id)

        # display the final response
        st.markdown(step_output.dict()["output"].response)

        st.subheader("Reasoning:")
        with st.expander("Show Reasoning"):
            # Display the intermediate reasoning steps
            for step in agent.get_completed_tasks()[-1].extra_state[
                "current_reasoning"
            ]:
                for key, value in step.dict().items():
                    if key not in ("return_direct", "action_input", "is_streaming"):
                        st.markdown(f"<span style='color: darkblue; font-weight: bold;'>{key}</span>: {value}", unsafe_allow_html=True)
                st.markdown("----")
