import os, json
import tempfile
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from utils import get_my_openai_key, format_user_question
from data_preprocessing import clone_repo, process_all_files
from model_context import ask_question, Context

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
    with tempfile.TemporaryDirectory() as repo_path:
        if clone_repo(github_url, repo_path):
            index, documents, file_type_counts, filenames = process_all_files(repo_path)
            if index is None:
                print("No Documents were found in the repo.")
                exit()
            print("Repository CLoned. Indexing files..")
            llm = OpenAI(api_key = MY_OPENAI_KEY)

            template = """
            Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {docs} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}

            Instructions:
            1. Answer based on files.
            2. Focus on repo and code.
            3. Purpose and features - describe purpose of repo.
            4. Functions/code - explain the function details and provide file name to reference code snippet.
            5. Setup/reuse of code - give instructions on how to use this code.

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
