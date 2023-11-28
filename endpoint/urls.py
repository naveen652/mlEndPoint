from django.urls import path
from .views import sentimentAnalysis

app_name = 'endpoint'

urlpatterns = [
    path('sentimentAnalysis/<str:email>', sentimentAnalysis, name='sentimentAnalysis'),
    path('', health, name='health')
]
