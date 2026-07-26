from django.urls import path
from . import views

urlpatterns = [
    path('patient-info/', views.patient_info, name='patient_info'),
    path('select-analysis/', views.select_analysis, name='select_analysis'),
]
