from django.urls import path
from .views import RegisterUser,GetPrediction,CheckFile,FileDownload

urlpatterns = [
    path('register/', RegisterUser.as_view()),
    path('upload/',GetPrediction.as_view()),
    path('checkfile/',CheckFile.as_view()),
    path('download/<str:username>/',FileDownload.as_view()),
]
