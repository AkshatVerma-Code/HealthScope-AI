from django import forms
from .models import PatientSession


class PatientForm(forms.ModelForm):
    class Meta:
        model = PatientSession
        fields = ['full_name', 'age', 'gender', 'contact', 'height', 'weight', 'blood_group', 'smoking_history', 'hypertension', 'heart_disease', 'diabetes', 'blood_infection']
        widgets = {
            'full_name': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'Enter your full name',
                'id': 'id_full_name',
            }),
            'age': forms.NumberInput(attrs={
                'class': 'form-input',
                'placeholder': 'Enter your age',
                'min': 1,
                'max': 120,
                'id': 'id_age',
            }),
            'gender': forms.Select(attrs={
                'class': 'form-select',
                'id': 'id_gender',
            }),
            'contact': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'Phone number (optional)',
                'id': 'id_contact',
            }),
            'hypertension': forms.CheckboxInput(attrs={'class': 'form-checkbox', 'id': 'id_hypertension'}),
            'heart_disease': forms.CheckboxInput(attrs={'class': 'form-checkbox', 'id': 'id_heart_disease'}),
            'diabetes': forms.CheckboxInput(attrs={'class': 'form-checkbox', 'id': 'id_diabetes'}),
            'blood_infection': forms.CheckboxInput(attrs={'class': 'form-checkbox', 'id': 'id_blood_infection'}),
            'height': forms.NumberInput(attrs={
                'class': 'form-input',
                'placeholder': 'Height in cm (optional)',
                'min': 50,
                'max': 300,
                'id': 'id_height',
            }),
            'weight': forms.NumberInput(attrs={
                'class': 'form-input',
                'placeholder': 'Weight in kg (optional)',
                'min': 10,
                'max': 300,
                'id': 'id_weight',
            }),
            'blood_group': forms.Select(attrs={
                'class': 'form-select',
                'id': 'id_blood_group',
            }),
            'smoking_history': forms.Select(attrs={
                'class': 'form-select',
                'id': 'id_smoking_history',
            }),
        }
        labels = {
            'full_name': 'Full Name',
            'age': 'Age',
            'gender': 'Gender',
            'contact': 'Contact Number',
            'hypertension': 'High Blood Pressure (Hypertension)',
            'heart_disease': 'Heart Disease',
            'diabetes': 'History of Diabetes',
            'blood_infection': 'Recent Blood Infection',
            'height': 'Height (cm)',
            'weight': 'Weight (kg)',
            'blood_group': 'Blood Group',
            'smoking_history': 'Smoking History',
        }
