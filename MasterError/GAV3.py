import streamlit as st
import google.generativeai as genai
from streamlit_monaco import st_monaco

# Configure the API key
genai.configure(api_key="Your_Gemini_API")

# Shared generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize models
error_detector_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=(
        "Assume you are a code generator:\n"
        "- Keep outputs simple and organized.\n"
        "- If user inputs any code, detect the code language and give suggestions for possible errors.\n"
        "- Provide correction code using optimized functions or classes.\n"
        "- Always include potential errors and improvements.\n"
        "- Summarize the output and include examples with both functions and classes and some .\n"
        "- if user input any code \ndetect the code language and give suggestions of possible errors and give correction code using optimized funx=ctions or classes\n- give us possible errors\n- and give the improvemnet in functions OR classses always for every code \n- organized and summarized the output in simple way and also include one example with function and one with class\n- more simple it and short it\n- need not to print Next Steps:\n\nI am ready to provide more details on more complex code.\n If programming language doesnot support function or class then show the best way to implement it with other languages. and give a simplest summary why use this language.\nI am ready to give potential error and improvements for complex code.\n"
    ),
)

executor_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=(
        "Assume you are a universal code executor:\n"
        "- Detect the programming language and display it.\n"
        "- Execute the code and show the output.\n"
        "- If errors occur, then give output of the error."
        "- if error comes include potential errors and improvements.\n"

    ),
)

# Start chat sessions
error_chat_session = error_detector_model.start_chat()
executor_chat_session = executor_model.start_chat()

# Initialize session state
if "analysis_response" not in st.session_state:
    st.session_state.analysis_response = None

if "execution_responses" not in st.session_state:
    st.session_state.execution_responses = []

# Streamlit UI
st.title("Code Analyzer and Executor")

# Error detection and improvement
st.header("Code Error Detector and Improver")
input_code = st_monaco(
    language="python", 
    theme="vs-dark", 
    height=400,
)

if st.sidebar.button("Analyze Code"):
    if input_code.strip():
        try:
            response = error_chat_session.send_message(input_code)
            st.session_state.analysis_response = response.text  # Store the analysis result
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please enter code for analysis.")

# Display analysis result
if st.session_state.analysis_response:
    st.markdown("### Analysis Result:")
    st.markdown(st.session_state.analysis_response)



# Button to clear execution history
if st.sidebar.button("Clear History"):
    st.session_state.execution_responses = []  # Clear the history
    st.sidebar.success("Execution history cleared.")

if st.sidebar.button("Execute Code", key="execute_code_button"):
    if input_code.strip():
        try:
            response_exec = executor_chat_session.send_message(input_code)
            st.session_state.execution_responses.append(response_exec.text)  # Store response
        except Exception as e:
            st.sidebar.error(f"Error during execution: {str(e)}")
    else:
        st.sidebar.warning("Please enter code for execution.")

# Display all execution responses
if st.session_state.execution_responses:
    st.sidebar.markdown("### Execution Results:")
    for i, res in enumerate(st.session_state.execution_responses, start=1):
        st.sidebar.markdown(f"**Attempt {i}:**\n{res}")
