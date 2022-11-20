from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import UserSerializer
from django.contrib.auth.models import User
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.authtoken.models import Token


class RegisterUser(APIView):
    def post(self,request):
        serializer = UserSerializer(data = request.data)

        if not serializer.is_valid():
            print(serializer.errors)
            return Response({'status':403,'errors':serializer.errors,'message':'Something went wrong'})

        serializer.save()
        user = User.objects.get(username = serializer.data['username'])
        token_obj , _= Token.objects.get_or_create(user = user)
        return Response({'status':200,'payload':serializer.data,'token':str(token_obj),'message':'your data is saved successfully'})
