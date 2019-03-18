from django.contrib import admin
from django.urls import path, include
from mlapi_app import views

urlpatterns = [
    path('hexa_point', views.HexaPoint.as_view()),
]
