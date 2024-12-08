from django.shortcuts import render

def index(request):
    return render(request, 'pages/index.html')

def upload(request):
    return render(request, 'pages/upload.html')
