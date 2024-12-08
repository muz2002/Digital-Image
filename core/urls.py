from django.urls import path, include
from core import views

urlpatterns = [
    path('', views.index, name='index'),
    path('', include('image_app.urls')),
]
