from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
from PIL import Image
from .forms import ImageUploadForm
from .utils_cnn import load_model, run_inference, MODEL_DESCRIPTIONS


@csrf_exempt
def model_inference(request, model_name):
    processed_image_url = None
    cnn_description = MODEL_DESCRIPTIONS.get(model_name, "No description available.")
    cnn_results = None
    error_message = None

    model = load_model(model_name)
    if model is None:
        error_message = "Model not found or not supported."

    if request.method == "POST" and model is not None:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data["image"]
            try:
                img = Image.open(image)
            except Exception:
                error_message = "Invalid image file."
                img = None

            if img:
                # Save uploaded image
                output_path = os.path.join(settings.MEDIA_ROOT, "uploaded.png")
                try:
                    img.save(output_path)
                    processed_image_url = settings.MEDIA_URL + "uploaded.png"
                except Exception:
                    error_message = "Error saving image."

                # Run inference
                if not error_message:
                    try:
                        cnn_results = run_inference(model, output_path, model_name)
                    except Exception as e:
                        error_message = f"Error during model inference: {str(e)}"
        else:
            error_message = "Invalid form submission."
    else:
        form = ImageUploadForm()

    return render(
        request,
        "pages/model_inference.html",
        {
            "form": form,
            "processed_image_url": processed_image_url,
            "cnn_description": cnn_description,
            "cnn_results": cnn_results,
            "error_message": error_message,
            "model_name": model_name,
        },
    )
