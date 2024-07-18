import os
import streamlit as st
import pickle
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_parse.base import ResultType

import nest_asyncio
nest_asyncio.apply()

import importlib
from src import tools, prompt_temp
importlib.reload(tools)
importlib.reload(prompt_temp)
from src.tools import get_nodes, calculate_circumferential_stress, decrypt_data
from src.prompt_temp import react_system_prmopt

from dotenv import load_dotenv
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

load_dotenv(override=True)

# Set the page layout to wide
st.set_page_config(layout="wide")

# Authentication configuration
AUTH_NAMES = os.getenv("AUTH_NAMES").split(",")
AUTH_USERNAMES = os.getenv("AUTH_USERNAMES").split(",")
AUTH_PASSWORDS = os.getenv("AUTH_PASSWORDS").split(",")

hashed_passwords = Hasher(AUTH_PASSWORDS).generate()

credentials = {
    "usernames": {
        AUTH_USERNAMES[0]: {
            "name": AUTH_NAMES[0],
            "password": hashed_passwords[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "cookie_name", "signature_key", cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login(fields=['username', 'password'])

if authentication_status:
    st.sidebar.success(f"Welcome {name}")

    # Load the API keys
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Set the embeddings and generative models
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    llm = OpenAI(model="gpt-4o")

    Settings.embed_model = embed_model
    Settings.llm = llm

    # Load the binary version of section 4 of CSA Z662
    loaded_documents_gpt4o = None

    # Try to load from FILE_PATH
    if os.getenv("FILE_PATH"):
        try:
            with open(os.getenv("FILE_PATH"), 'rb') as f:
                loaded_documents_gpt4o = pickle.load(f)
            print("File loaded successfully from FILE_PATH")
        except (FileNotFoundError, IOError, pickle.PickleError) as e:
            print(f"Error loading from FILE_PATH: {e}")

    # If FILE_PATH failed, try ENCRYPTED_FILE_PATH
    if loaded_documents_gpt4o is None and os.getenv("ENCRYPTED_FILE_PATH"):
        try:
            encrypted_file_path = os.getenv("ENCRYPTED_FILE_PATH")
            Binary_file_password = os.getenv("BINARY_FILE_PASSWORD")
            
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted = decrypt_data(encrypted_data, Binary_file_password)
            loaded_documents_gpt4o = pickle.loads(decrypted)
            print("File decrypted and loaded successfully from ENCRYPTED_FILE_PATH")
        except (FileNotFoundError, IOError, ValueError) as e:
            print(f"Error loading from ENCRYPTED_FILE_PATH: {e}")

    # If both previous attempts failed, try ORIGINAL_FILE_PATH
    if loaded_documents_gpt4o is None and os.getenv("ORIGINAL_FILE_PATH"):
        try:
            parser_gpt4o = LlamaParse(
                result_type=ResultType.MD,
                gpt4o_mode=True,
            )
            loaded_documents_gpt4o = parser_gpt4o.load_data(os.getenv("ORIGINAL_FILE_PATH"))
            print("File loaded successfully from ORIGINAL_FILE_PATH")
            data = pickle.dumps(loaded_documents_gpt4o)

            with open(os.getenv("FILE_PATH"), 'wb') as f:
                f.write(data)
            
        except Exception as e:
            print(f"Error loading from ORIGINAL_FILE_PATH: {e}")

    # Check if any loading method was successful
    if loaded_documents_gpt4o is not None:
        print("Documents loaded successfully")
    else:
        print("Failed to load documents from any source")

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
            with st.expander("Show Progress"):
                step_output = agent.run_step(task.task_id)
                st.markdown(step_output.dict()["output"].response)

                # Check whether the task is complete
                while step_output.is_last == False:
                    step_output = agent.run_step(task.task_id)
                    st.markdown(step_output.dict()["output"].response)

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

elif authentication_status == False:
    st.error("Username/password is incorrect")

elif authentication_status == None:
    st.warning("Please enter your username and password")
