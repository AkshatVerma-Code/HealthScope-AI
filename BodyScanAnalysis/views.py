from django.shortcuts import render, redirect, get_object_or_404
from .models import MedicalImage
from .dl_predict import predict_image, MODEL_CONFIGS
from UserRegistration.models import PatientSession


def image_upload(request):
    """Medical image upload page."""
    session_id = request.session.get('patient_session_id')
    if not session_id:
        return redirect('patient_info')

    patient = get_object_or_404(PatientSession, session_id=session_id)

    IMAGE_CHOICES = [
        ('BRAIN_MRI', 'Brain MRI — Tumor / Glioma / Meningioma'),
        ('CHEST_XRAY', 'Chest X-ray — Pneumonia / Tuberculosis'),
    ]

    if request.method == 'POST':
        image_type = request.POST.get('image_type', 'BRAIN_MRI')
        uploaded_file = request.FILES.get('medical_image')

        if not uploaded_file:
            return render(request, 'image_upload.html', {
                'patient': patient,
                'image_choices': IMAGE_CHOICES,
                'error': 'Please upload an image file.',
            })

        # Save record first
        medical_image = MedicalImage.objects.create(
            session=patient,
            image_type=image_type,
            uploaded_file=uploaded_file,
        )

        # Run DL inference
        try:
            result = predict_image(medical_image.uploaded_file.path, image_type)
            medical_image.prediction = result.get('prediction', 'Unknown')
            medical_image.confidence = result.get('confidence', 0.0)
            medical_image.all_class_scores = result.get('all_class_scores', {})
            medical_image.model_available = result.get('model_available', False)
            medical_image.save()
        except Exception as e:
            medical_image.prediction = 'Prediction failed'
            medical_image.confidence = 0.0
            medical_image.model_available = False
            medical_image.save()

        return redirect('image_result', image_id=medical_image.id)

    return render(request, 'image_upload.html', {
        'patient': patient,
        'image_choices': IMAGE_CHOICES,
    })


def image_result(request, image_id):
    """Medical image result page."""
    medical_image = get_object_or_404(MedicalImage, id=image_id)
    session_id = request.session.get('patient_session_id')

    if str(medical_image.session.session_id) != str(session_id):
        return redirect('patient_info')

    # Prepare class score list for chart rendering
    class_scores = [
        {'label': cls, 'score': score}
        for cls, score in medical_image.all_class_scores.items()
    ]

    return render(request, 'image_result.html', {
        'medical_image': medical_image,
        'patient': medical_image.session,
        'class_scores': class_scores,
    })
