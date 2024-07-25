import streamlit as st
from llama_index.core import Settings
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# import importlib
# from src import tools, prompt_temp
# importlib.reload(tools)
# importlib.reload(prompt_temp)
from src.tools import MARKDOWN_QUERY_ENGINE
from src.prompt_temp import qa_system_prompt, react_system_prmopt
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

from llama_index.llms.openai import OpenAI as llma_OpenAI
llm = llma_OpenAI(model="gpt-4o")
Settings.llm = llm

@ st.cache_resource()
def ReAct_Agent():

    gpt_4o = OpenAIMultiModal(model="gpt-4o", max_new_tokens=200)
    query_engine = MARKDOWN_QUERY_ENGINE(
        qa_prompt = qa_system_prompt(), multi_modal_llm=gpt_4o
    )
    
    pipe_design_guideline_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="pipe_design_guideline",
        description=(
            "This is a guideline for the design of brined steel pipelines."
        ),
    )
    agent = ReActAgent.from_tools(
        [pipe_design_guideline_tool], llm=llm, verbose=True
    )
    # Load a custom system prompt
    react_system_prompt = react_system_prmopt()

    # Update the agent with the custom system prompt
    agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

    # Reset the agent
    agent.reset()
    return agent

st.title("ALA2005 Pipeline Design Guideline Augmented with a ReAct Agent")

query_str = 'How to calculate axial stiffness of soil springs?'
query = st.text_input("**Enter your query:**", query_str)

# Create a task
agent = ReAct_Agent()
task = agent.create_task(query)

# Iterate over the thought, action, and observation steps to complete the task
if st.button("Submit"):
    with st.spinner("Processing your query..."):
        with st.expander("Show Progress"):
            step_output = agent.run_step(task.task_id)
            st.markdown(step_output.dict()["output"].response)

            # Check whether the task is complete
            while step_output.is_last == False:
                step_output = agent.run_step(task.task_id)
                st.markdown(step_output.dict()["output"].response)

        # display the final response
        st.subheader("Final Answer:")
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