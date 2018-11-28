from .models import Hexagon
from rest_framework import serializers

class HexagonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Hexagon
        fields = ('member_num','data_file','ml_result')