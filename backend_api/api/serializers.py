from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Results

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username','password']

    def create(self,validated_data):
        user = User.objects.create(username = validated_data['username'])
        user.set_password(validated_data['password'])
        user.save()
        return user

class FileUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Results
        fields = ['username','gene_file']
