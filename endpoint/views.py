import pandas as pd
import requests
from django.http import JsonResponse
from rest_framework.decorators import api_view
import paddle
#import nltk
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
#nltk.download('vader_lexicon')
#from nltk.sentiment import SentimentIntensityAnalyzer
# final model


def query(payload, API_URL, headers):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

@api_view(['GET'])
def sentimentAnalysis(request, email, id):
    response = requests.get('https://mindwellnesspro.onrender.com/userresponse/'+email)
    d = response.json()
    if not d:
        return JsonResponse({"response":"no data available"})
    #df_json = pd.DataFrame.from_dict(data)
    else:
        name=d[0]['name']
        email=d[0]['email']
        suggestions="suggestions from genAI"
        responses_data = d[0]['responses']
        if responses_data:
            unique_id_data = requests.post('https://mindwellnesspro.onrender.com/reports/'+email+"/")
            unique_id_response = unique_id_data.json()
            unique_id=unique_id_response['UniqueId']
            responses_df = pd.DataFrame(responses_data)
            questions_data = d[0]['questions']
            response_text = responses_df['response'].str.cat(sep='. ')
            if(id==0):
	    	API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
	    else if(id==1):
		API_URL = "https://api-inference.huggingface.co/models/arpanghoshal/EmoRoBERTa"
	    else:
		return JsonResponse({'error':'invalid id, choose id 0 for specific test and 1 for neutral test'})
            headers = {"Authorization": "Bearer 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"}
            output = query({"inputs": response_text,}, API_URL, headers)
            sentiment=output[0][0]['label']
            sentiment_score=output[0][0]['score']
            result_data={'unique id':unique_id,'name':name,'email':email,'suggestions':suggestions, "sentiment":sentiment, "score": sentiment_score, "status":1}
            return JsonResponse(result_data)
        else:
            result_data={'unique id':unique_id,'name':name,'email':email,'sentiment':'no sentiment', 'score':0,'suggestions':suggestions}
            return JsonResponse(result_data)
@api_view(['GET'])
def health(request):
    return JsonResponse({'health':'healthy'})
