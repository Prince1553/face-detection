 
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('single_image_search/', views.single_image_search_view, name='single_image_search'),
    path('group_image_search/', views.group_image_search_view, name='group_image_search'),
    path('download_zip_file/', views.download_zip_file, name='download_zip_file'),
]