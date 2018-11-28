from rest_framework import serializers
from .models import Hexagon

class HexagonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Hexagon
        fields = ('data_file','ml_result')

    def create(self, validated_data):
        return Hexagon.objects.create(**validated_data)
    
class HexaTestSerailizer(serializers.ModelSerializer):
    
    #file_url = serializers.SerializerMethodField('get_file_url')
    
    class Meta:
        model = Hexagon
        fields = ('data_file',)

   # def get_file_url(self, obj):
   #    return obj.data_file.url