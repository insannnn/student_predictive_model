from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Student, Course, Enrollment, Assessment, Attendance, Department
from .forms import GradePredictionForm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def home(request):
    # Hitung statistik dasar untuk halaman beranda
    student_count = Student.objects.count()
    course_count = Course.objects.count()
    enrollment_count = Enrollment.objects.count()
    
    context = {
        'student_count': student_count,
        'course_count': course_count, 
        'enrollment_count': enrollment_count
    }
    return render(request, 'academic_analytics/home.html', context)

def about(request):
    return render(request, 'academic_analytics/about.html')

def dashboard(request):
    # Dapatkan semua departemen
    departments = Department.objects.all()
    
    # Jumlah siswa per departemen
    dept_student_counts = []
    for dept in departments:
        count = Student.objects.filter(dept=dept).count()
        dept_student_counts.append({'name': dept.dept_name, 'count': count})
    
    # Rata-rata nilai per departemen
    dept_avg_grades = []
    for dept in departments:
        students = Student.objects.filter(dept=dept)
        enrollments = Enrollment.objects.filter(stu__in=students)
        # Hanya ambil yang tidak kosong
        valid_grades = [e.grade for e in enrollments if e.grade is not None]
        if valid_grades:
            avg_grade = sum(valid_grades) / len(valid_grades)
        else:
            avg_grade = 0
        dept_avg_grades.append({'name': dept.dept_name, 'avg_grade': avg_grade})
    
    context = {
        'dept_student_counts': dept_student_counts,
        'dept_avg_grades': dept_avg_grades
    }
    return render(request, 'academic_analytics/dashboard.html', context)

def admin_panel(request):
    prediction_result = None
    
    if request.method == 'POST':
        form = GradePredictionForm(request.POST)
        if form.is_valid():
            # Dapatkan data dari form
            student = form.cleaned_data['student']
            course = form.cleaned_data['course']
            attendance = form.cleaned_data['attendance']
            midterm_score = form.cleaned_data['midterm_score']
            project_score = form.cleaned_data['project_score']
            
            # Proses prediksi menggunakan model machine learning
            prediction_result = predict_grade(student, course, attendance, midterm_score, project_score)
    else:
        form = GradePredictionForm()
    
    context = {
        'form': form,
        'prediction_result': prediction_result
    }
    return render(request, 'academic_analytics/admin_panel.html', context)

def predict_grade(student, course, attendance, midterm_score, project_score):
    # Dapatkan data untuk training
    enrollments = Enrollment.objects.all()
    data = []
    
    for enrollment in enrollments:
        # Dapatkan nilai ujian jika ada
        midterm = Assessment.objects.filter(enroll=enrollment, assessment_type='Midterm').first()
        final = Assessment.objects.filter(enroll=enrollment, assessment_type='Final').first()
        project = Assessment.objects.filter(enroll=enrollment, assessment_type='Project').first()
        
        # Dapatkan kehadiran jika ada
        attend = Attendance.objects.filter(enroll=enrollment).first()
        
        # Hanya tambahkan jika semua data tersedia dan nilai akhir (grade) tidak null
        if midterm and final and project and attend and enrollment.grade is not None:
            data.append({
                'midterm_score': midterm.score,
                'final_score': final.score,
                'project_score': project.score,
                'attendance': attend.attendance_percentage,
                'grade': enrollment.grade
            })
    
    # Jika tidak ada data yang cukup
    if len(data) < 10:
        return {
            'predicted_grade': 'Not enough data for prediction',
            'confidence': 0
        }
    
    # Convert ke DataFrame
    df = pd.DataFrame(data)
    
    # Pisahkan fitur dan target
    X = df[['midterm_score', 'project_score', 'attendance']]
    y = df['grade']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Buat dan latih model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Input untuk prediksi
    input_data = np.array([[midterm_score, project_score, attendance]])
    input_data_scaled = scaler.transform(input_data)
    
    # Prediksi
    predicted_grade = model.predict(input_data_scaled)[0]
    
    # Hitung confidence (bisa menggunakan std dev dari prediksi ensemble)
    predictions = []
    for estimator in model.estimators_:
        predictions.append(estimator.predict(input_data_scaled)[0])
    
    confidence = 1 - (np.std(predictions) / 100)  # Penskalaan sederhana
    
    return {
        'predicted_grade': round(predicted_grade, 2),
        'confidence': round(confidence * 100, 2)
    }