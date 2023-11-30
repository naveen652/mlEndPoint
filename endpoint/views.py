import pandas as pd
import requests
from django.http import JsonResponse
from rest_framework.decorators import api_view
import paddle
import nltk
from transformers import pipeline
#nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
# final model


def analyze_sentiment_emoroberta(text):
    sentiment_analyzer = pipeline("sentiment-analysis",'arpanghoshal/EmoRoBERTa')
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Example usage
text_to_analyze = "iam happy"
sentiment_label, sentiment_score = analyze_sentiment(text_to_analyze)

def analyze_sentiment(text):
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Get the sentiment scores
    sentiment_scores = sid.polarity_scores(text)

    # Classify the sentiment
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    compound_score=abs(sentiment_scores['compound']*100)
    return sentiment, compound_score

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
        suggestions="suggestions from genAI"
        responses_data = d[0]['responses']
        if responses_data:
            unique_id_data = requests.post('https://mindwellnesspro.onrender.com/reports/'+email+"/")
            unique_id_response = unique_id_data.json()
            unique_id=unique_id_response['UniqueId']
            responses_df = pd.DataFrame(responses_data)
            questions_data = d[0]['questions']
            response_text = responses_df['response'].str.cat(sep='. ')
            sentiment, sentiment_score = analyze_sentiment_emoroberta(response_text)
            result_data={'unique id':unique_id,'name':name,'email':email,'suggestions':suggestions, "sentiment":sentiment, "score": sentiment_score, "status":1}
            return JsonResponse(result_data)
        else:
            result_data={'unique id':unique_id,'name':name,'email':email,'sentiment':'no sentiment', 'score':0,'suggestions':suggestions}
            return JsonResponse(result_data)
@api_view(['GET'])
def health(request):
    return JsonResponse({'health':'healthy'})
