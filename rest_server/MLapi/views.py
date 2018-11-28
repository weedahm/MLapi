from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse
from rest_framework import viewsets, status
from .serializers import HexagonSerializer, HexaTestSerailizer
from .models import Hexagon

from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, detail_route, list_route
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import generics

from MLapi import learningFunction
from MLapi.ikzziML.common import calcFunctions


class hexa_test(generics.ListCreateAPIView):
    serializer_class = HexaTestSerailizer
    queryset = Hexagon.objects.all()

    def get(self, request, *args, **kwargs):
        if 'patient_id' in self.request.query_params:

            #file_path = HexaTestSerailizer.get_file_url(self, 'patient_id')

            #file_path = Hexagon.objects.get(pk=self.request.query_params.get('patient_id'))
            file_path = 'C:/Users/KimTaeWoo/MLapi/rest_server/bodychart.json'

            MODEL_PATH = 'C:/Users/KimTaeWoo/MLapi/rest_server/MLapi/patients_2layerNN/2layerNN.ckpt'
            DATA_PREPRO_PATH = 'C:/Users/KimTaeWoo/MLapi/rest_server/MLapi/patients_2layerNN/dataPreprocessing.csv'
            predict_data = learningFunction.supervised_learning_inference(file_path, MODEL_PATH, DATA_PREPRO_PATH)
            predict_score = calcFunctions.setScore(predict_data)

            print(file_path)

        return Response(predict_score)


