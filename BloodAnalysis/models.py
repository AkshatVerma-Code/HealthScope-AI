from django.db import models
from UserRegistration.models import PatientSession


class BloodReport(models.Model):
    REPORT_TYPE_CHOICES = [
        ('CBC', 'Complete Blood Count'),
        ('LFT', 'Liver Function Test'),
        ('KFT', 'Kidney Function Test'),
        ('SUGAR', 'Blood Sugar Report'),
        ('HBA1C', 'HbA1c Report'),
        ('COMBINED', 'Combined Reports'),
    ]

    session = models.ForeignKey(
        PatientSession, on_delete=models.CASCADE, related_name='blood_reports'
    )
    report_type = models.CharField(max_length=10, choices=REPORT_TYPE_CHOICES)
    uploaded_file = models.FileField(upload_to='blood_reports/', blank=True, null=True)
    extracted_params = models.JSONField(default=dict, blank=True)
    predictions = models.JSONField(default=dict, blank=True)
    recommendations = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session.full_name} — {self.get_report_type_display()} ({self.created_at.date()})"

    class Meta:
        ordering = ['-created_at']
