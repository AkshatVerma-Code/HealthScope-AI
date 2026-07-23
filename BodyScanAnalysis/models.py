from django.db import models
from UserRegistration.models import PatientSession


class MedicalImage(models.Model):
    IMAGE_TYPE_CHOICES = [
        ('BRAIN_MRI', 'Brain MRI'),
        ('CHEST_XRAY', 'Chest X-ray'),
    ]

    session = models.ForeignKey(
        PatientSession, on_delete=models.CASCADE, related_name='medical_images'
    )
    image_type = models.CharField(max_length=20, choices=IMAGE_TYPE_CHOICES)
    uploaded_file = models.ImageField(upload_to='medical_images/')
    prediction = models.CharField(max_length=200, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    all_class_scores = models.JSONField(default=dict, blank=True)
    model_available = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session.full_name} — {self.get_image_type_display()} ({self.prediction})"

    class Meta:
        ordering = ['-created_at']
