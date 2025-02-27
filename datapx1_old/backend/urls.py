from django.urls import path,include
from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('api/gptresponse/', gpt_response, name='gptresponse'),
    path('api/gpt_graphical/', gpt_graphical, name='gpt_graphical'),
    path('api/file_upload/', uploadFile, name='file_upload'),
    path('api/dataprocess', data_processing, name='data_process'),
    path('api/kpi_process', kpi_prompt, name="kpi_process"),
    path('api/mvt', mvt, name='mvt'),
    path('api/generate_code', kpi_code, name="kpi_code"),
    path(r'api/models', models, name='models'),
    path('api/model_predict', model_predict, name='model_predict'),
    path('api/genai_bot', gen_ai_bot, name='visual_ai_bot'),
]
