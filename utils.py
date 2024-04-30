import json
import os
#from dotenv import load_dotenv
from openai import OpenAI
import re
import nltk

nltk.download("punkt")

def get_my_openai_key():
    path = "c:/shared/content/config/api-keys"
    os.chdir(path)
    openai_keys = json.load(open('hackathon_openai_keys.json'))
    my_openai_key = openai_keys['team_23']
    #print(my_openai_key)
    #openai_keys = json.load(open('hackathon_openai_keys.json'))
    #my_openai_key = openai_keys['team_23']
    return my_openai_key

#This reguler expression are copied using google regex generator
def text_regex_builder(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return nltk.word_tokenize(text) #this method is inbuilt methond of NLP 

#this are helper functions used for any NLP projects where all documents / chunks are combined and indexed each one as a unique document
def format_documents(documents):
    numbered_docs = "\n".join([f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content}" for i, doc in enumerate(documents)])
    return numbered_docs

def format_user_question(question):
    question = re.sub(r'\s+', ' ', question).strip()
    return question
