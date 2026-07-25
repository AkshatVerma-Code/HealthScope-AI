from django.urls import path
from . import views

urlpatterns = [
    path('blood-report/upload/', views.blood_upload, name='blood_upload'),
    path('blood-report/result/<int:report_id>/', views.blood_result, name='blood_result'),
    path('blood-report/download/<int:report_id>/', views.blood_download_pdf, name='blood_download_pdf'),
    path('blood-report/manual/', views.manual_entry, name='manual_entry'),
]
