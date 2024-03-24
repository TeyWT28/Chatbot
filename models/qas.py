import numpy as np
import pandas as pd
import os
from serpapi import GoogleSearch
from Levenshtein import ratio
from pathlib import Path

p = Path(__file__)

"""###**Data Pre-processing**"""

# Read Q/A pairs from files

df_08 = pd.read_csv(p.with_name('S08_question_answer_pairs.txt'), sep='\t')
df_09 = pd.read_csv(p.with_name('S09_question_answer_pairs.txt'), sep='\t')
df_10 = pd.read_csv(p.with_name('S10_question_answer_pairs.txt'), sep='\t', encoding = 'ISO-8859-1')

# Combine three files
df_all = df_08.append([df_09, df_10])

# Get question and answer pairs only
df_all_1 = df_all[['Question', 'Answer']]

# Drop rows with NaN/null values
df_all_1 = df_all_1.dropna(axis=0)

# Drop duplicate questions
df_all_2 = df_all_1.drop_duplicates(subset='Question')


"""###**QAS Model**"""

def getResult(question, func):
  answer, prediction = func(question)
  return [question, prediction, answer]


def getApproximateAnswer(q):
  max_score = 0
  answer = ""
  prediction = ""
  for idx, row in df_all_2.iterrows():
    score = ratio(row["Question"], q)
    if score >= 0.9:
      print("Existing Q/A Pair")
      if score == 1:  # exact match
        return row["Answer"], "Q/A bank"
      else:
        return 'Did you mean <i>"' +row["Question"] + '"</i> <br>' + 'Answer: ' +row["Answer"], "Q/A bank"
    elif score > max_score:  # find the highest score in the Q/A bank
      max_score = score
      answer = row["Answer"]
      prediction = row["Question"]
  if max_score > 0.6:
    print("max_score:", max_score)
    return 'Do you mean <i>"' +prediction + '"</i> <br>' + 'Answer: ' +answer, "Q/A bank"
  else:  # if question not found in question bank, google it
    answer, ans_type = gg_search(q)
    if ans_type == "answer_box":   
      return answer, "Google Search"
    elif ans_type == "organic_results":   
      sen = answer['snippet']
      s = sen[:sen.rfind('. ')]  # Extract until last complete sentence
      res = s + '<br><br> Find out more here: ' + '<a href=' +answer['link'] +' target="_blank">' +answer['title'] +'</a>'
      return res, "Google Search"
    else:
      return answer, "no answer"


def gg_search(query):
  params = {
    "api_key": "0b2296bcb8580b3abde4c7e25be501e145f5371b5d96b587fe5f2cafbb0eaea3",
    "engine": "google",
    "q":  query,
  }
  search = GoogleSearch(params)
  results = search.get_dict()
  try:  
    print("Answer box")   # Direct answer is found
    answer = results['answer_box']['answer']
    return answer, "answer_box"
  except:   
    pass  # Fall back to organic results
  try:
    print("Organic results")  # Return ans from first search result
    answer = results['organic_results'][0]
    return answer, "organic_results"
  except:   # nothing is found
    return "Sorry, I was unable to find any result :(", "none"


def QAS(query):
  result = getResult(query, getApproximateAnswer)
  return result

