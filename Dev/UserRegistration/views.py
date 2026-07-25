from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.urls import reverse
from .forms import PatientForm
from .models import PatientSession
from BloodAnalysis.models import BloodReport
from BodyScanAnalysis.models import MedicalImage


def patient_info(request):
    """Step 1: Collect patient information."""
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save()
            # Store session_id in Django session
            request.session['patient_session_id'] = str(patient.session_id)
            request.session['patient_name'] = patient.full_name

            # Store blood pressure for ML prediction.
            # If hypertension is checked and no BP entered, default to 120.
            bp_raw = request.POST.get('blood_pressure', '').strip()
            if bp_raw:
                request.session['blood_pressure'] = int(bp_raw)
            elif patient.hypertension:
                request.session['blood_pressure'] = 120
            else:
                request.session['blood_pressure'] = None
            
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect_url': reverse('select_analysis')})
            return redirect('select_analysis')
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'errors': form.errors}, status=400)
    else:
        form = PatientForm()

    return render(request, 'patient_info_enchanced.html', {'form': form})


def select_analysis(request):
    """Step 2: Choose between blood report or image analysis."""
    session_id = request.session.get('patient_session_id')
    if not session_id:
        return redirect('patient_info_enchanced')

    patient = get_object_or_404(PatientSession, session_id=session_id)
    return render(request, 'select_analysis_enhanced.html', {'patient': patient})


def dashboard(request, session_id):
    """Full session summary dashboard."""
    patient = get_object_or_404(PatientSession, session_id=session_id)
    blood_reports = BloodReport.objects.filter(session=patient).order_by('-created_at')
    medical_images = MedicalImage.objects.filter(session=patient).order_by('-created_at')

    context = {
        'patient': patient,
        'blood_reports': blood_reports,
        'medical_images': medical_images,
    }
    return render(request, 'dashboard.html', context)
