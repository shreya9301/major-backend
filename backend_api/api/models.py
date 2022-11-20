from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Results(models.Model):
    resultid = models.AutoField(primary_key=True)
    username = models.ForeignKey(User,default = None)
    path_to_data = models.CharField(unique = True,blank = None)
    prediction = models.CharField(blank = None)

    def __str__(self):
        return self.username