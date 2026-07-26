from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from .models import BloodReport
from .ocr import extract_parameters
from .ml_predict import run_predictions, generate_recommendations
from .pdf_report import generate_pdf
from UserRegistration.models import PatientSession


def blood_upload(request):
    """Blood report upload page."""
    session_id = request.session.get('patient_session_id')
    if not session_id:
        return redirect('patient_info_enchanced')

    patient = get_object_or_404(PatientSession, session_id=session_id)

    REPORT_CHOICES = [
        ('CBC', 'Complete Blood Count (CBC)'),
        ('LFT', 'Liver Function Test (LFT)'),
        ('KFT', 'Kidney Function Test (KFT)'),
    ]

    if request.method == 'POST':
        params = {}
        processed_files = 0
        primary_report_type = None

        for value, label in REPORT_CHOICES:
            uploaded_file = request.FILES.get(f'report_file_{value}')
            if uploaded_file:
                if not primary_report_type:
                    primary_report_type = value
                processed_files += 1
                try:
                    extracted = extract_parameters(uploaded_file, value)
                    params.update(extracted)
                except Exception as e:
                    pass

        report_type = 'COMBINED' if processed_files > 1 else (primary_report_type or 'COMBINED')

        # ── DEBUG: Print extracted params to terminal ─────────────────────────
        print("\n" + "="*60)
        print(f"[DEBUG] RAW OCR EXTRACTED PARAMETERS — Report Type: {report_type}")
        print("="*60)
        if params:
            for k, v in params.items():
                print(f"  {k:<35} = {v}")
        else:
            print("  (no parameters extracted — OCR may have failed)")
        print("="*60 + "\n")
        # ─────────────────────────────────────────────────────────────────────

        # Run ML predictions
        predictions = run_predictions(params, report_type, patient) if params else {}

        # Generate recommendations
        recommendations = generate_recommendations(predictions)

        # Save results without the physical file
        report = BloodReport.objects.create(
            session=patient,
            report_type=report_type,
            extracted_params=params,
            predictions=predictions,
            recommendations=recommendations,
        )

        return redirect('blood_result', report_id=report.id)

    return render(request, 'blood_upload.html', {
        'patient': patient,
        'report_choices': REPORT_CHOICES,
    })


def blood_result(request, report_id):
    """Blood report results page."""
    report = get_object_or_404(BloodReport, id=report_id)
    session_id = request.session.get('patient_session_id')

    # Security: only show report if it belongs to current session
    if str(report.session.session_id) != str(session_id):
        return redirect('patient_info')

    # ── DEBUG: Print stored params + patient info to terminal ──────────────
    patient = report.session
    print("\n" + "="*60)
    print(f"[DEBUG] BLOOD RESULT VIEW — Report #{report.id} ({report.report_type})")
    print("-"*60)
    print(f"  Patient     : {patient.full_name}, Age {patient.age}, {patient.gender}")
    print(f"  Hypertension: {patient.hypertension}")
    print(f"  Blood Infect: {patient.blood_infection}")
    print(f"  Sugar Level : {patient.sugar}")
    print("-"*60)
    print("[DEBUG] STORED EXTRACTED PARAMS:")
    if report.extracted_params:
        for k, v in report.extracted_params.items():
            print(f"  {k:<35} = {v}")
    else:
        print("  (empty)")
    print("-"*60)
    print("[DEBUG] ML PREDICTIONS:")
    for disease, data in report.predictions.items():
        print(f"  {disease:<20} risk={data.get('risk')}%  status={data.get('status')}  available={data.get('available')}")
    print("="*60 + "\n")
    # ─────────────────────────────────────────────────────────────────────

    # Prepare risk data for gauge charts
    risk_data = []
    for disease, data in report.predictions.items():
        if data.get('available') and data.get('risk') is not None:
            risk_data.append({
                'disease': disease,
                'risk': data['risk'],
                'status': data['status'],
            })

    return render(request, 'blood_result.html', {
        'report': report,
        'patient': report.session,
        'risk_data': risk_data,
    })


def blood_download_pdf(request, report_id):
    """Generate and download PDF report."""
    report = get_object_or_404(BloodReport, id=report_id)
    session_id = request.session.get('patient_session_id')

    if str(report.session.session_id) != str(session_id):
        return redirect('patient_info')

    try:
        pdf_bytes = generate_pdf(report)
        response = HttpResponse(pdf_bytes, content_type='application/pdf')
        filename = f"LabSense_Report_{report.session.full_name.replace(' ', '_')}_{report.id}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        return HttpResponse(f"PDF generation failed: {str(e)}", status=500)


def manual_entry(request):
    """Manual parameter entry (fallback when no OCR file)."""
    session_id = request.session.get('patient_session_id')
    if not session_id:
        return redirect('patient_info')

    patient = get_object_or_404(PatientSession, session_id=session_id)

    if request.method == 'POST':
        report_type = request.POST.get('report_type', 'COMBINED')

        # Collect all numeric POST params
        params = {}
        for key, val in request.POST.items():
            if key in ('csrfmiddlewaretoken', 'report_type'):
                continue
            try:
                params[key] = float(val)
            except (ValueError, TypeError):
                if val.lower() in ('yes', 'true', '1'):
                    params[key] = 1
                elif val.lower() in ('no', 'false', '0'):
                    params[key] = 0

        predictions = run_predictions(params, report_type, patient)
        recommendations = generate_recommendations(predictions)

        report = BloodReport.objects.create(
            session=patient,
            report_type=report_type,
            extracted_params=params,
            predictions=predictions,
            recommendations=recommendations,
        )

        return redirect('blood_result', report_id=report.id)

    return render(request, 'blood_upload.html', {
        'patient': patient,
        'manual_mode': True,
    })
