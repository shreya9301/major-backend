from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import *


# class MyUserAdmin(UserAdmin):
#         model = User
#         list_display = ('email','id','username',
#                         'first_name','is_active','is_staff')
#         list_filter = ('username',
#                         'first_name','last_name')
#         ordering = ('email', )
#         filter_horizontal = ()
#         fieldsets = (None, {'fields': ('username', 'password',)}),
#         ('Personal info', {                       # Here
#             'fields': ('first_name', 'last_name',)}),
#         ('Permissions', {'fields': ('is_active','is_staff', 'is_superuser',
#                                        'groups', 'user_permissions',)}),
#         ('Important dates', {'fields': ('last_login', 'date_joined',)})

admin.site.register(User)
admin.site.register(Results)
