# import streamlit as st
# from benchmark import get_pipeline
# from utils import prepare_document
# import pandas as pd
# from hip_agent import HIPAgent


# st.title("MCQ taker")


# def start_app():
#     return HIPAgent()

# st.sidebar.title("Prompt Type")

# prompt_dict = {"Zero-Shot": "General QnA", "Few-Shot": "General QnA", "Chain of thought": "Generic QnA with Reasoning", "Retrieval Augmented Generation": "Domain Specific QnA with Reasoning"}

# selected_page = st.sidebar.radio('Menu', prompt_dict)

# st.header(selected_page)
# st.subheader("Usecase: " + prompt_dict[selected_page]) 
 
# if selected_page == "Retrieval Augmented Generation":
    
#         # add select box for using existing files or uploading new files
#         file_type = st.selectbox("Select Source", ("Use Existing Sources", "Upload New Sources"))
#         st.write("You selected", file_type)
    
#         if file_type == "Use Existing Sources":
#             st.write("The textbook being used is:")
    
#         elif file_type == "Upload New Sources":
#             txt_files = st.file_uploader("Upload txt file", type=["txt"])
#             d_submit = st.button("Submit")

#             if txt_files is not None and d_submit:
#                 with st.spinner("processing file…"):
#                     df = prepare_document(txt_files)


# # upload file to run benchmark
# st.subheader("Run Benchmark")
# data = st.file_uploader("Upload a csv file", type=["csv"])

# benchmark = st.button("Run Benchmark")

# if data is not None and benchmark:
#     with st.spinner("Running benchmarks. Please hold..."):
#         score, answers = get_pipeline(data, selected_page )
#         st.write(f"Score: {score}/{len(answers)}")

# st.subheader("Input your own question")       
# question = st.text_input("Enter your question here 💭 ")
# option_A = st.text_input("Enter option A here 💭 ")
# option_B = st.text_input("Enter option B here 💭 ")
# option_C = st.text_input("Enter option C here 💭 ")
# option_D = st.text_input("Enter option D here 💭 ")

# q_button = st.button("Answer")
# if question != "" and all([option_A, option_B, option_C, option_D]) and q_button:
#     with st.spinner("Thinking. Please hold..."):
#         agent = start_app()
#         answer_index, reasoning = agent.get_response(question, [option_A, option_B, option_C, option_D], selected_page, return_reasoning=True)
#         st.write(f"Answer: {chr(answer_index + ord('a'))}")
#         st.write(f"Reasoning: {reasoning}")

import streamlit as st
from benchmark import get_pipeline
from utils import prepare_document
import pandas as pd
from hip_agent import HIPAgent
import traceback

st.title("MCQ taker")

def start_app():
   try:
    return HIPAgent()
   
   except Exception as e:
        print("An error occurred while starting the app. Details:")
        print(traceback.format_exc())

def configure_sidebar():
    st.sidebar.title("Prompt Type")
    prompt_dict = {"Zero-Shot": "General QnA", "Few-Shot": "General QnA", "Chain of thought": "Generic QnA with Reasoning", "Retrieval Augmented Generation": "Domain Specific QnA with Reasoning"}
    selected_page = st.sidebar.radio('Menu', prompt_dict)
    st.header(selected_page)
    st.subheader("Usecase: " + prompt_dict[selected_page]) 
    return selected_page

def prepare_document_source():
    if selected_page == "Retrieval Augmented Generation":
        file_type = st.selectbox("Select Source", ("Use Existing Sources", "Upload New Sources"))
        st.write("You selected", file_type)
        if file_type == "Use Existing Sources":
            st.write("The textbook being used is:")
        elif file_type == "Upload New Sources":
            txt_files = st.file_uploader("Upload txt file", type=["txt"])
            d_submit = st.button("Submit")
            if txt_files is not None and d_submit:
                with st.spinner("processing file…"):
                    df = prepare_document(txt_files)

def run_benchmark():
    st.subheader("Run Benchmark")
    data = st.file_uploader("Upload a csv file", type=["csv"])
    benchmark = st.button("Run Benchmark")
    if data is not None and benchmark:
        with st.spinner("Running benchmarks. Please hold..."):
            score, answers = get_pipeline(data, selected_page )
            st.write(f"Score: {score}/{len(answers)}")

def input_question():
    st.subheader("Input your own question")       
    question = st.text_input("Enter your question here 💭 ")
    option_A = st.text_input("Enter option A here 💭 ")
    option_B = st.text_input("Enter option B here 💭 ")
    option_C = st.text_input("Enter option C here 💭 ")
    option_D = st.text_input("Enter option D here 💭 ")
    q_button = st.button("Answer")
    if question != "" and all([option_A, option_B, option_C, option_D]) and q_button:
        with st.spinner("Thinking. Please hold..."):
            agent = start_app()
            answer_index, reasoning = agent.get_response(question, [option_A, option_B, option_C, option_D], selected_page, return_reasoning=True)
            st.write(f"Answer: {chr(answer_index + ord('a'))}")
            st.write(f"Reasoning: {reasoning}")

selected_page = configure_sidebar()
prepare_document_source()
run_benchmark()
input_question()
