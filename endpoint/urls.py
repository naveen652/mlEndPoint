from django.urls import path
from .views import *

app_name = 'endpoint'

urlpatterns = [
    path('sentimentAnalysis/<int:id>/<str:email>', sentimentAnalysis, name='sentimentAnalysis'),
    path('', health, name='health')
]
