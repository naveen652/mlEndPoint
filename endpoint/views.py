import pandas as pd
import requests
from django.http import JsonResponse
from rest_framework.decorators import api_view
from transformers import pipeline


# final model
@api_view(['GET'])
def sentimentAnalysis(request, email):
    response = requests.get('https://mindwellnesspro.onrender.com/userresponse/'+email)
    d = response.json()
    if not d:
        return JsonResponse({"response":"no data available"})
    #df_json = pd.DataFrame.from_dict(data)
    else:
        name=d[0]['name']
        email=d[0]['email']
        unique_id_data = requests.post('https://mindwellnesspro.onrender.com/reports/'+email+"/")
        unique_id_response = unique_id_data.json()
        unique_id=unique_id_response['UniqueId']
        suggestions="suggestions from genAI"
        responses_data = d[0]['responses']
        if not responses_data:
            result_data={'unique id':unique_id,'name':name,'email':email,'sentiment':'no sentiment', 'score':0,'suggestions':suggestions}
            return JsonResponse(result_data)
        responses_df = pd.DataFrame(responses_data)
        questions_data = d[0]['questions']
        #questions_df = pd.DataFrame(questions_data)
        #concatinated text of questions and responses
        #questions_text = questions_df['Question'].str.cat(sep='\n')
        response_text = responses_df['response'].str.cat(sep='\n ')
        #list of questions and responses
        #list_of_questions = questions_df['Question'].tolist()
        #list_of_responses = responses_df['response'].tolist()
        sentiment_analyzer = pipeline("sentiment-analysis",'arpanghoshal/EmoRoBERTa')
        result = sentiment_analyzer(response_text)
        sentiment_label=result[0]['label']
        sentiment_score=result[0]['score']
        result_data={'unique id':unique_id,'name':name,'email':email,'sentiment':sentiment_label, 'score':sentiment_score*100,'suggestions':suggestions}
        return JsonResponse(result_data)
