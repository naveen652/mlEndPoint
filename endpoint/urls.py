from django.urls import path
from .views import sentimentAnalysis

app_name = 'endpoint'

urlpatterns = [
    path('sentimentAnalysis/', sentimentAnalysis, name='sentimentAnalysis')
]
