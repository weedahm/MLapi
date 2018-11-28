from django.contrib import admin
from MLapi.models import Hexagon

class HexaAdmin(admin.ModelAdmin):
    # list_display = ('data_file','ml_result',)
     list_display = ('data_file',)

admin.site.register(Hexagon, HexaAdmin)