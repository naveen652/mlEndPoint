import pandas as pd
import requests
from django.http import JsonResponse
from rest_framework.decorators import api_view
import paddle
import pprint
import google.generativeai as palm
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

def query(payload, API_URL, headers):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

@api_view(['GET'])
def sentimentAnalysis(request, id, email):
  response = requests.get('https://mindwellnesspro.onrender.com/userresponse/'+email)
  d = response.json()
  if not d:
    return JsonResponse({"error: no data available"})
    #df_json = pd.DataFrame.from_dict(data)
  else:
    name=d[0]['name']
    email=d[0]['email']
    responses_data = d[0]['responses']
    if responses_data:
      palm.configure(api_key='AIzaSyACs8z3ksFw7CKmiPDFEpxDZ3Rhw4vymRM')
      unique_id_data = requests.post('https://mindwellnesspro.onrender.com/reports/'+email+"/")
      unique_id_response = unique_id_data.json()
      unique_id=unique_id_response['UniqueId']
      responses_df = pd.DataFrame(responses_data)
      questions_data = d[0]['questions']
      questions_df = pd.DataFrame(questions_data)
      category=questions_df['Category'][0]
      list_of_questions = questions_df['Question'].tolist()
      list_of_responses = responses_df['response'].tolist()
      list_of_texts = responses_df['text'].tolist()
      description_text=responses_df['text'].str.cat(sep='. ')
      response_text = responses_df['response'].str.cat(sep='. ')
      input=response_text+description_text
      if(id==0):
        API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
      elif(id==1):
        API_URL = "https://api-inference.huggingface.co/models/arpanghoshal/EmoRoBERTa"
      else:
        return JsonResponse({'error':'invalid id, choose id 0 for specific test and 1 for neutral test'})
      headers = {"Authorization": "Bearer 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"}
      output = query({"inputs": input,},API_URL, headers)
      sentiments_scores=output
      positive=sentiments_scores[0][0]['score']
      neutral=sentiments_scores[0][1]['score']
      negative=sentiments_scores[0][2]['score']
      prompt = "hi,give me some suggestions to improve my mental health and Also discuss about my scores and responses with respected questions. I have taken mental health assesment on "+category+", evaluated by machine learning model.analyse my answers to questions and individual scores i got in the assesment and provide me with some suggestions so that i can improve my mental health. In that assesment i got positive score of "+str(positive)+", negative score of "+str(negative)+"and neutral score of "+str(neutral)+"out of 100%. these are the list of questions in the assesment:" +str(list_of_questions)+ "and these are the list of responses i have given for those list of questions:"+str(list_of_responses)+"and these are the list of descriptions i have given for those list of questions:"+str(list_of_texts)+"please consider my list of responses and also my list of descriptions for giving suggestions. "
      completion = palm.chat(messages=prompt)
      suggestions=completion.last
      result_data={'unique id':unique_id,'name':name,'email':email,'suggestions':suggestions, "sentiments_scores": sentiments_scores, "status":1}
      return JsonResponse(result_data)
    else:
      result_data={'name':name,'email':email, 'error':'no responses'}
      return JsonResponse(result_data)@api_view(['GET'])
def health(request):
	return JsonResponse({'health':'server is up'})

@api_view(['GET'])
def healthly(request):
	return JsonResponse({'health':'healthy'})
