from django.urls import path
from . import views

urlpatterns = [
    path('image-analysis/upload/', views.image_upload, name='image_upload'),
    path('image-analysis/result/<int:image_id>/', views.image_result, name='image_result'),
]
