from django.db import models

class Hexagon(models.Model):
    data_file = models.FileField(null=True)
    # ml_result = models.CharField(max_length=50)