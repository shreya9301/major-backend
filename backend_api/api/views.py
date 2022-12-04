from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import viewsets
from .serializers import UserSerializer
from rest_framework.exceptions import AuthenticationFailed
from .models import User,Results
from .fileHandle import handle_uploaded_file
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated,AllowAny
from .utility import predict
from rest_framework.parsers import FileUploadParser,JSONParser,FormParser,MultiPartParser
from rest_framework.authtoken.models import Token
import jwt
import datetime
import os
from wsgiref.util import FileWrapper
from django.http import HttpResponse


class RegisterUser(APIView):
    permission_classes = [AllowAny]
    parser_classes = [JSONParser,FormParser,MultiPartParser]
    def post(self, request):
        serializer = UserSerializer(data=request.data)

        if not serializer.is_valid():
            print(serializer.errors)
            return Response({'status': 403, 'errors': serializer.errors, 'message': 'Something went wrong'})

        serializer.save()
        user = User.objects.get(username=serializer.data['username'])
        return Response({'status': 200, 'payload': serializer.data, 'message': 'your data is saved successfully'})


class GetPrediction(APIView):
   # permission_classes = [IsAuthenticated,]
    def post(self,request):

        username = request.data.get('username')
        user = User.objects.get(username=username)
        parser_classes = [FileUploadParser]
        gene_data = request.data['gene_file']

        final_path = handle_uploaded_file(user,gene_data)
        print("-------------------------------------------" + final_path)
        with open(final_path, 'w') as destination:
            for chunk in gene_data.chunks():
                destination.write(str(chunk))

        GeneObj = Results(username = user,gene_data_path = final_path,date_uploaded = datetime.datetime.now().date())
        GeneObj.save()
        print("saved")
        #predict
        gene_filename = gene_data.name
        #result_file_path = predict(final_path)

        return Response({'status':200,'message':'The gene_file is saved successfully'})

class CheckFile(APIView):
    def post(self,request):
        username = request.data.get('username')
        userObj = User.objects.get(username = username)
        resultObj = Results.objects.get(username = userObj)
        #print(resultObj)
        result_file_path = resultObj.gene_data_path
        required_path = result_file_path[:-4] + "_results.csv"
        print(required_path)
        if (os.path.isfile(required_path) == True):
            return HttpResponse(status = 200)
        else:
            return HttpResponse(status = 404)
        
        
class FileDownload(APIView):

    def get(self, request,username):
        userObj = User.objects.get(username = username)
        resultObj = Results.objects.get(username = userObj)
        result_file_path = resultObj.gene_data_path
        required_path = result_file_path[:-4] + "_results.csv"
        print(required_path)
        
        with open(str(required_path), 'r') as file:
            response = HttpResponse(file, content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=file.csv'
            return response