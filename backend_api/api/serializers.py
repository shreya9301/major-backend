from rest_framework import serializers
from .models import User,Results

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('first_name','last_name','email','username','password')
        extra_kwargs = {'password':{'required':False}}

    def create(self,validated_data):
        password = validated_data.pop('password',None)
        instance = self.Meta.model(**validated_data)
        instance.is_active = True
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance

# class FileUploadSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Results
#         fields = ['username','gene_data_path']
