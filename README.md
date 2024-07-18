# CSA-Z662 Pipe Design Standard Assistant

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genai-applications-in-pipeline-engineering.streamlit.app/).

This Streamlit application assists users in retrieving specific information from the CSA Z662 pipe design standard and performing calculations related to pipeline engineering. The app leverages Retrieval-Augmented Generation (RAG) using the OpenAI model for question answering and contextual information retrieval.

## Features

- **Retrieve Information from CSA Z662**: Get detailed context from the CSA Z662 pipe design standard.
- **Calculate Circumferential Stress**: Compute the circumferential (hoop) stress in a pipeline based on the CSA Z662 standard.
- **Interactive User Interface**: Simple and intuitive interface powered by Streamlit.
- Other features will be added.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Farhad-Davaripour/GenAI_Applications_in_Pipeline_Engineering.git
    cd GenAI_Applications_in_Pipeline_Engineering
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Environment Variables**:
    Create a `.env` file from .env_template and add the following:
    ```plaintext
    LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
    OPENAI_API_KEY=your_openai_api_key
    DEFAULT_QUERY=your_default_query
    ```

5. **Run the Application**:
    ```bash
    streamlit run demo.py
    ```

## File Descriptions

- **demo.py**: Main script for the Streamlit application. It sets up the environment, loads models, and defines the app layout and functionality.
- **src/tools.py**: Contains utility functions for processing documents and performing calculations.
- **src/prompt_temp.py**: Defines custom prompt templates for the ReAct system.

## Usage

1. **Enter Query**: Input your query in the text area provided.
2. **Submit Query**: Click on the "Submit" button to process your query.
3. **View Results**: The app will display the final response along with the reasoning steps.

## Deployed Version

You can access the deployed version of the Streamlit app [here](https://genai-applications-in-pipeline-engineering.streamlit.app/).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
