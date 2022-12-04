from django.db import models
from django.utils import timezone
from django.contrib.auth.base_user import BaseUserManager
from django.contrib.auth.models import PermissionsMixin,AbstractBaseUser
from django.utils.translation import gettext_lazy as _
from django.conf import settings
from .fileHandle import handle_uploaded_file


class CustomUserManager(BaseUserManager):

    def create_superuser(self, email, username, first_name,last_name,password, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError(_('Superuser must have is_staff=True.'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_('Superuser must have is_superuser=True.'))
        return self.create_user(email, username, first_name,last_name,password, **extra_fields)

    def create_user(self, email, username,first_name,last_name,password, **extra_fields):
        email = self.normalize_email(email=email)
        if not email:
            raise ValueError(_('The email must be set'))
        user = self.model(email = email,username=username,first_name=first_name,last_name=last_name, **extra_fields)
        user.set_password(password)
        user.save()
        return user


# Create your models here.

class User(AbstractBaseUser,PermissionsMixin):
    email = models.EmailField(_('email address'),unique = True)
    username = models.CharField(max_length=100,unique=True)
    first_name = models.CharField(max_length = 100,blank = True)
    last_name = models.CharField(max_length = 100, blank = True)
    start_date = models.DateTimeField(default = timezone.now)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    objects = CustomUserManager()
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username','first_name','last_name']

    def __str__(self):
        return self.username


class Results(models.Model):
    username = models.ForeignKey(settings.AUTH_USER_MODEL,default = None,on_delete = models.CASCADE)
    gene_data_path = models.CharField(max_length = 150,primary_key = True, blank = True)
    date_uploaded = models.DateField()
    #prediction = models.CharField(max_length = 10,blank = True)


