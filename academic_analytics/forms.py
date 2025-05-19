from django import forms
from .models import Student, Course, Enrollment

class GradePredictionForm(forms.Form):
    student = forms.ModelChoiceField(queryset=Student.objects.all(), required=True)
    course = forms.ModelChoiceField(queryset=Course.objects.all(), required=True)
    attendance = forms.IntegerField(min_value=0, max_value=100, required=True)
    midterm_score = forms.IntegerField(min_value=0, max_value=100, required=True)
    project_score = forms.IntegerField(min_value=0, max_value=100, required=True)