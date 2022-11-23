from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import viewsets
from .serializers import UserSerializer,FileUploadSerializer
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth.models import User
from .models import Results
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.authtoken.models import Token
import jwt
import datetime


class RegisterUser(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)

        if not serializer.is_valid():
            print(serializer.errors)
            return Response({'status': 403, 'errors': serializer.errors, 'message': 'Something went wrong'})

        serializer.save()
        user = User.objects.get(username=serializer.data['username'])
        token_obj, _ = Token.objects.get_or_create(user=user)
        return Response({'status': 200, 'payload': serializer.data, 'token': str(token_obj), 'message': 'your data is saved successfully'})


class LoginUser(APIView):
    def post(self, request):
        user = request.data.get('username')
        password = request.data.get('password')

        user = User.objects.filter(username=user).first()

        if user is None:
            raise AuthenticationFailed('User not found!')

        if not user.check_password(password):
            raise AuthenticationFailed('Incorrect password!')

        payload = {
            'id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
            'iat': datetime.datetime.utcnow()
        }

        token = jwt.encode(payload, 'secret',
                           algorithm='HS256')

        response = Response()

        response.set_cookie(key='jwt', value=token, httponly=True)
        response.data = {
            'jwt': token
        }
        return response


class LogoutUser(APIView):
    def post(self, request):
        response = Response()
        response.delete_cookie('jwt')
        response.data = {
            'message': 'success'
        }
        return response


class GetPrediction(APIView):
   # permission_classes = [IsAuthenticated,]
    def post(self,request):
        # serializer = FileUploadSerializer(data = request.data)
        # if serializer.is_valid():
        #     serializer.save()  

        username = request.data.get('username')
        user = User.objects.get(username=username)
        gene_data = request.FILES.get('gene_file')
        GeneObj = Results(username = user,gene_data = gene_data)
        GeneObj.save()
        # content_type = gene_data.content_type
        #prediction = get_cancer_prediction()
        # response = "POST API and you have uploaded a {} file".format(content_type)
        return Response({'status':200,'message':'The gene_file is saved successfully'})

