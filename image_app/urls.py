from django.urls import path
from .views import model_inference

urlpatterns = [
    path('model/<str:model_name>/', model_inference, name='model_inference'),
]
