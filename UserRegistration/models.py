from django.db import models
import uuid


class PatientSession(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other'),
    ]

    BLOOD_GROUP_CHOICES = [
        ('A+', 'A+'), ('A-', 'A-'), ('B+', 'B+'), ('B-', 'B-'),
        ('AB+', 'AB+'), ('AB-', 'AB-'), ('O+', 'O+'), ('O-', 'O-')
    ]

    SUGAR_LEVEL_CHOICES = [
        (0, 'No diabetes'),
        (1, 'Controlled diabetes'),
        (2, 'Moderately high sugar'),
        (3, 'High sugar'),
        (4, 'Very high sugar'),
        (5, 'Extremely high / uncontrolled'),
    ]

    session_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    full_name = models.CharField(max_length=200)
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    contact = models.CharField(max_length=20, blank=True)

    # Vitals & Lifestyle
    height = models.PositiveIntegerField(null=True, blank=True, help_text="Height in cm")
    weight = models.PositiveIntegerField(null=True, blank=True, help_text="Weight in kg")
    blood_group = models.CharField(max_length=5, choices=BLOOD_GROUP_CHOICES, blank=True)

    # Medical History (used for fallback if OCR doesn't extract them)
    hypertension = models.BooleanField(default=False, verbose_name="High Blood Pressure")
    blood_infection = models.BooleanField(default=False, verbose_name="Recent Blood Infection")
    sugar = models.IntegerField(
        default=0,
        choices=SUGAR_LEVEL_CHOICES,
        verbose_name="Sugar Level (0–5)"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.full_name} ({self.session_id})"

    class Meta:
        ordering = ['-created_at']
