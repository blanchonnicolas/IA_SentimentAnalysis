#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# #Save in root folder IABadBuzz_AirParadis_FastAPI.py
# #Launch in cmd line: uvicorn IABadBuzz_AirParadis_FastAPI:app --reload
# #Access swagger in http://127.0.0.1:8000/docs#/ through webbrowser


# In[3]:


import preprocessor as p
import numpy as np
import pandas as pd
#Access swagger in  http://localhost:8000/predict text="This app is a total waste of time, but provides sentiment analysis!"
from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

#from .classifier.model import Model, get_model
from transformers import pipeline, BertForSequenceClassification
import datasets

from joblib import dump, load

tokenizer=load('./Models/bert_tokenizer.joblib')
#model_bert1=load('./Models/model_bert1.joblib')
trainer = BertForSequenceClassification.from_pretrained('./Models/Bert_Trainer5')

app = FastAPI()
pipe = pipeline("sentiment-analysis")

# preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length
def preprocess_function(dataset):
    return tokenizer(dataset["text"], truncation=True, padding=True) #'max_length', return_tensors="tf"

#  DataCollatorWithPadding to create a batch of examples
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app, based on HuggingFace pipeline'}


@app.get('/sentiment_analysis/')
async def query_sentiment_analysis(text: str):
    text_series = pd.Series(text)
    text_df = pd.DataFrame(text_series,  columns = ['text'])
    text_dict = datasets.Dataset.from_pandas(test_text_df)
    tokenized_sentiment = text_dict.map(preprocess_function, batched=True)
# test_text_dict = datasets.Dataset.from_pandas(test_text_df)
# tokenized_test_text_dict = test_text_dict.map(preprocess_function, batched=True)
# test_text_encoding = tokenizer(test_text)
# print(test_text_encoding)
    return analyze_sentiment(tokenized_sentiment)


def analyze_sentiment(tokenized_sentiment):
    """Get and process result"""
    result=[]
    #result = pipe(text)
    y_prob = trainer.predict(tokenized_sentiment)
    y_pred = np.argmax(y_prob.predictions, axis=-1)
    for pred in y_pred:
        if pred == 0:
            result.append('NEGATIVE') 
        elif pred == 1:
            result.append('POSITIVE')

    # Format and return results
    return {'result': result}


# In[ ]:




