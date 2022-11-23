from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Results(models.Model):
    resultid = models.AutoField(primary_key=True)
    username = models.ForeignKey(User,default = None,on_delete = models.CASCADE)
    #path_to_data = models.CharField(max_length = 100,unique = True,blank = None)
    gene_file = models.FileField(upload_to="backend_api/media/",blank = True,null = True)
    prediction = models.CharField(max_length = 10,blank = None)

    def __str__(self):
        return self.username