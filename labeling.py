import pickle

with open('../data/df_chained.pkl', 'rb') as file:
    df_chained = pickle.load(file)

df_chained = df_chained.reset_index()
df_chained = df_chained.drop(columns=["level_0", "index"])

# df_chained = df_chained.head(10)

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm
tqdm.pandas()

template = """
You are an expert annotator that classifies texts as polite or non-polite.
Please classify the following text as either 'Polite' or 'Non-polite'.
Text: {email_body}
"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3:8b-instruct-q2_K")
chain = prompt | model

def annotate_sarcasm(email_body):
    response = chain.invoke({"email_body": email_body})
    cleaned_response = response.strip()

    if "Polite" in cleaned_response:
        return "Polite"
    elif "Non-polite" in cleaned_response:
        return "Non-polite"
    else:
        return None

df_chained['polite'] = df_chained['chained_main_body'].progress_apply(annotate_sarcasm)

import pickle

with open('../data/df_chained_with_polite.pkl', 'wb') as file:
    pickle.dump(df_chained, file)
