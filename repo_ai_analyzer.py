import os, json
import tempfile
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from utils import get_my_openai_key, format_user_question
from data_preprocessing import clone_repo, process_all_files
from model_context import ask_question, Context
import streamlit as st
import time

load_dotenv()
MY_OPENAI_KEY = get_my_openai_key()
WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
model_name = "gpt-3.5-turbo"


def main_handler(github_url):
    #github_url = input('Enter the Github URL:')
    repo_name = github_url.split("/")[-1]
    print("Cloning the repository...")
    st.sidebar.info("Cloning the repository...")
    with tempfile.TemporaryDirectory() as repo_path:
        if clone_repo(github_url, repo_path):
            index, documents, file_type_counts, filenames = process_all_files(repo_path)
            if index is None:
                print("No Documents were found in the repo.")
                exit()
            print("Repository CLoned. Indexing files..")
            st.sidebar.info("Repository Cloned Succesfully. Analyzing & Indexing files..")
            llm = OpenAI(api_key = MY_OPENAI_KEY)

            template = """
            Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {docs} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}

            Instructions:
            1. Answer based on context/docs.
            2. Focus on repo/code.
            3. Consider:
                a. Purpose/features - describe.
                b. Functions/code - provide details/samples.
                c. Setup/usage - give instructions.

            Answer:
            """

            prompt = PromptTemplate(
                template = template,
                input_variables=["repo_name","github_url","conversation_history","question","docs","file_type_counts","filenames"]
            )

            llm_chain = LLMChain(prompt=prompt,llm=llm)

            conversation_history = ""
            question_context = Context(index,documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
            return question_context
        else:
            print("Failed to clone the repository.")

def generate_response(user_question,question_context):
    try:
        #user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit:' to quit): " + RESET_COLOR)
        if user_question.lower() == "exit:":
            exit()
        print("Thinking...")
        user_question = format_user_question(user_question)

        answer = ask_question(user_question, question_context)
        #print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
        #conversation_history += f"Question: {user_question} \nAnswer: {answer}\n"
        return answer
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

@st.experimental_fragment
def main(q_c):
    #github_url = input('Enter the Github URL:')
    #q_c = main_handler(url)
    flag = True
    element_id = 0
    with st.form('my_form'):
        text = st.text_area('Ask questions to the agent:', '')
        submitted = st.form_submit_button('Submit')
        exitted = st.form_submit_button('Exit')
        if submitted:
            print("now generating reponse...")
            #st.info("Thinking..")
            progress_text = "Thinking... Please wait."
            my_bar = st.progress(0, text=progress_text)
            answer = generate_response(text,q_c)
            for percent_complete in range(100):
                time.sleep(0.3)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            st.success(answer)
        if exitted:
            st.info("Thank you for using this app.")
            exit()
    '''
    while True:
        if flag:
            element_id += 1
            flag = False
            #user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit:' to quit): " + RESET_COLOR)
            user_question = st.text_area('Enter text:', key=f"txt_{element_id}")
            if (st.button('Submit')):
                if user_question.lower() == 'exit:':
                    break
        else:
            answer = generate_response(user_question,q_c)
            st.info(answer)
            flag = True
        #print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
    print("THANK YOU")
'''

'''
if __name__ == "__main__":
    github_url = input('Enter the Github URL:')
    q_c = main_handler(github_url)
    while True:
        user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit:' to quit): " + RESET_COLOR)
        if user_question.lower() == 'exit:':
            break
        answer = generate_response(user_question,q_c)
        print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
    print("THANK YOU")
'''
