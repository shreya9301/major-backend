from django.urls import path
from .views import LoginUser, LogoutUser, RegisterUser

urlpatterns = [
    path('register/', RegisterUser.as_view()),
    path('login/', LoginUser.as_view()),
    path('logout/', LogoutUser.as_view())
]
