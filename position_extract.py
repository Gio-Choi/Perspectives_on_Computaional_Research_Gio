import pickle

with open('../data/df_chained.pkl', 'rb') as file:
    df_chained = pickle.load(file)

df_chained = df_chained.reset_index()
df_chained = df_chained.drop(columns=["level_0", "index"])


import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm
tqdm.pandas()

template = """
You are given the body of an email. The email may contain a job title after closing salutations and their name.

Extract only the sender's job title (position) from the email body. Do not rephrase or add any additional wording.
Then, determine whether they are senior level or not. 

Email body:
{email_body}

Output must follow this format (If there's no job title, return None for both):
  - Job Title: (job title)
  - Seniority: Yes or No 
  
"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3:8b-instruct-q2_K")
chain = prompt | model

def extract_position(email_body):
    response = chain.invoke({"email_body": email_body})
    cleaned_response = response.strip()
    
    if "Job Title:" in cleaned_response and "Seniority:" in cleaned_response:
        try:
            job_title_part = cleaned_response.split("Job Title:")[1]
            job_title = job_title_part.split("Seniority:")[0].strip()
        except IndexError:
            job_title = None
        
        try:
            seniority = cleaned_response.split("Seniority:")[1].strip()
        except IndexError:
            seniority = None

        return pd.Series([job_title if job_title else None, seniority if seniority else None])
    
    return pd.Series([None, None])

df_chained[['position', 'seniority']] = df_chained['chained_main_body'].apply(extract_position)

import pickle

with open('../data/df_chained_with_positions_llama3_instruct.pkl', 'wb') as file:
    pickle.dump(df_chained, file)
