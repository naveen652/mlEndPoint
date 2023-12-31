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
    return JsonResponse({"error":"no data available"})
    #df_json = pd.DataFrame.from_dict(data)
  else:
    name=d[0]['name']
    email=d[0]['email']
    responses_data = d[0]['responses']
    if responses_data:
      palm.configure(api_key='AIzaSyACs8z3ksFw7CKmiPDFEpxDZ3Rhw4vymRM')
      models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
      model = models[0].name
      unique_id_data = requests.post('https://mindwellnesspro.onrender.com/reports/'+email+"/")
      unique_id_response = unique_id_data.json()
      unique_id=unique_id_response['UniqueId']
      responses_df = pd.DataFrame(responses_data)
      questions_data = d[0]['questions']
      questions_df = pd.DataFrame(questions_data)
      list_of_questions = questions_df['Question'].tolist()
      list_of_responses = responses_df['response'].tolist()
      response_text = responses_df['response'].str.cat(sep='. ')
      if(id==0):
        API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
      elif(id==1):
        API_URL = "https://api-inference.huggingface.co/models/arpanghoshal/EmoRoBERTa"
      else:
        return JsonResponse({'error':'invalid id, choose id 0 for specific test and 1 for neutral test'})
      headers = {"Authorization": "Bearer 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"}
      output = query({"inputs": response_text,},API_URL, headers)
      sentiments_scores=output
      positive=sentiments_scores[0][0]['score']
      neutral=sentiments_scores[0][1]['score']
      negative=sentiments_scores[0][2]['score']
prompt = f"Hi, I've recently completed a mental health assessment focusing on 'depression' and received scores indicating Positive: {positive}, Negative: {negative}, and Neutral: {neutral} out of 100%. I'm seeking guidance and suggestions to enhance my mental well-being based on my assessment. Here's a summary of the assessment: {list_of_questions}. My responses to these questions were: {list_of_responses}."
 

completion = palm.generate_text(model=model,prompt=prompt)
      suggestions=completion.result
      result_data={'unique id':unique_id,'name':name,'email':email,'suggestions':suggestions, "sentiments_scores": sentiments_scores[0], "status":1}
      return JsonResponse(result_data)
    else:
      result_data={'name':name,'email':email, 'error':'no responses'}
      return JsonResponse(result_data)
@api_view(['GET'])
def health(request):
	return JsonResponse({'health':'server is up'})

@api_view(['GET'])
def healthly(request):
	return JsonResponse({'health':'healthy'})
