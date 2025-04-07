#==================Setup==================#
# Import necessary libraries and modules
import streamlit as st

# Llama Index core and OpenAI multi-modal LLM dependencies
from llama_index.core import Settings
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# Reload custom modules after import
# import importlib
# from src import tools, prompt_temp
# importlib.reload(tools)
# importlib.reload(prompt_temp)

# Import specific tools and prompt templates from custom modules
from src.tools import MARKDOWN_QUERY_ENGINE, calculate_axial_soil_force
from src.prompt_temp import qa_system_prompt, react_system_prompt as RA_SYSTEM_PROMPT

# Import tools and agent framework from Llama Index
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent

# Import OpenAI LLM wrapper from Llama Index
from llama_index.llms.openai import OpenAI as llma_OpenAI

# Loading environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize the LLM and set it in Settings
llm_type = st.sidebar.selectbox("Select LLM type", ["gpt-4o", "o1-preview"])
llm = llma_OpenAI(model="gpt-4o")

multi_modal_llm = OpenAIMultiModal(model=llm_type, max_new_tokens=400)
query_engine = MARKDOWN_QUERY_ENGINE(
    qa_prompt = qa_system_prompt(), multi_modal_llm=multi_modal_llm
)

#=====================tool#1========================#
#ALA2005 design guideline
pipe_design_guideline_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="ALA_pipe_design_guideline",
    description=(
        "Provides access to the ALA Guidelines for the Design of Buried Steel Pipe. "
        "Covers design provisions for evaluating buried steel pipelines under various loads, including internal pressure, earth loads, buoyancy, "
        "thermal expansion, and seismic effects. Includes hand calculation equations and finite element analysis guidance. "
        "Focused on delivering information from this guideline."
    )
)

#=====================tool#2========================#
# Calculate axial soil force
axial_soil_force_tool = FunctionTool.from_defaults(
    fn=calculate_axial_soil_force,
    name="axial_soil_force_calculator",
    description="Calculates the maximum axial soil force per unit length of a buried pipeline based on the American Lifelines Alliance (ALA) 2005 guidelines. This tool performs the calculation using the pipe outside diameter, soil cohesion, depth to pipe centerline, effective unit weight of soil, adhesion factor, coefficient of earth pressure at rest, and the interface friction angle between the pipe and soil. For detailed explanations, derivations, and the full context of the equation, refer to the relevant sections of the ALA guidelines."
)

# Initialize the ReAct Agent and pass the predefined tools
agent = ReActAgent.from_tools(
    [pipe_design_guideline_tool, axial_soil_force_tool], 
    llm=llm, verbose=True
)

# Load a custom system prompt
react_system_prompt = RA_SYSTEM_PROMPT()

# Update the agent with the custom system prompt
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

st.title("Pipe Design Agent")
st.markdown("""This application assists Pipeline engineers in retrieving specific information from the ALA2005 pipe design standard 
        ([American Lifelines Alliance 2005: Guidelines for the Design of Buried Steel Pipe](https://www.americanlifelinesalliance.com/pdf/Update061305.pdf))
        and performing calculations related to the different sections of the guideline. The app leverages a ReAct (Reasoning and Acting) Agentic
        Workflow to split the user query into smaller pieces and execute them sequentially or in parallel to answer.
        It utilizes `Retrieval Augmented Generation (RAG)` for question answering and contextual information retrieval and `Function Calling` to perform calculations.""")

query_str = """1. How to calculate the maximum axial soil spring force on a buried pipeline using ALA design guidelines? 
2. Calculate the maximum axial soil force for a 35-inch buried pipeline at a burial depth of 1.5 m. Assume cohesion of 25 kpa and fiction angle of 20 deg. Ensure using correct units for each variable.
"""

query = st.text_area("**Enter your query:**", query_str)

# Create a task
task = agent.create_task(query)

# Iterate over the thought, action, and observation steps to complete the task
if st.button("Submit"):
    with st.spinner("Processing your query..."):
        with st.expander("Show Progress"):
            step_output = agent.run_step(task.task_id)
            st.markdown(step_output.dict()["output"]["response"])

            # Check whether the task is complete
            while step_output.is_last == False:
                step_output = agent.run_step(task.task_id)
                st.markdown(step_output.dict()["output"]["response"])

        # display the final response
        st.subheader("Final Answer:")
        st.markdown(step_output.dict()["output"]["response"])

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