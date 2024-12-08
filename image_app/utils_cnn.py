# image_app/utils_cnn.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

MODEL_DESCRIPTIONS = {
    'resnet50': "ResNet50: Deep residual network with 50 layers.",
    'vgg16': "VGG16: A 16-layer CNN architecture with simple structure.",
    'mobilenet_v2': "MobileNetV2: Lightweight model for mobile devices.",
    'inception_v3': "InceptionV3: Inception modules for efficiency.",
    'googlenet': "GoogLeNet: Inception-based architecture by Google.",
    'lenet': "LeNet trained on MNIST for handwritten digit recognition (0-9).",
    'yolov5': "YOLOv5: Object detection model by Ultralytics."
}

BASE_DIR = Path(__file__).resolve().parent.parent

class LeNetMNIST(nn.Module):
    def __init__(self):
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # Input: 1x28x28, Output: 6x24x24
        self.pool = nn.AvgPool2d(2, 2)                 # Output: 6x12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Output: 16x8x8
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*4*4)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=False, aux_logits=False)
    elif model_name == 'lenet':
        # Load the trained LeNet model
        model = LeNetMNIST()
        weights_path = BASE_DIR / 'image_app' / 'lenet_mnist.pt'  # Update extension to .pt
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            model.eval()
            print("LeNet model loaded successfully.")
        except Exception as e:
            print(f"Error loading LeNet model: {e}")
            return None
    elif model_name == 'yolov5':
        # YOLOv5 from ultralytics hub
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    else:
        model = None

    if model is not None and model_name != 'yolov5':
        model.eval()
    return model

# Preprocessing transforms for LeNet (MNIST)
mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Preprocessing transforms for ImageNet models
classification_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet classes
imagenet_classes_path = BASE_DIR / 'image_app' / 'imagenet_classes.txt'
with open(imagenet_classes_path) as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

def run_inference(model, img_path, model_name):
    if model_name == 'yolov5':
        # YOLOv5 inference
        results = model(img_path)
        det = results.xyxy[0]
        top = min(5, det.size(0))
        yolo_results = []
        for i in range(top):
            c = int(det[i,5].item())
            conf = det[i,4].item()
            label = results.names[c]
            yolo_results.append((label, conf))
        return yolo_results

    elif model_name == 'lenet':
        # LeNet (MNIST) inference
        img = Image.open(img_path).convert('RGB')
        img = mnist_transform(img)
        img = img.unsqueeze(0)  # [1,1,28,28]

        with torch.no_grad():
            output = model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_class = probabilities.max(dim=0)
        predicted_digit = top_class.item()
        return [(str(predicted_digit), top_prob.item())]

    else:
        # ImageNet classification models
        img = Image.open(img_path).convert('RGB')
        input_tensor = classification_transform(img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
        if isinstance(output, tuple):
            output = output[0]

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        results = []
        for i in range(top5_prob.size(0)):
            idx = top5_catid[i].item()
            if idx < len(imagenet_classes):
                class_name = imagenet_classes[idx]
                probability = top5_prob[i].item()
                results.append((class_name, probability))
            else:
                results.append(("Unknown", top5_prob[i].item()))
        return results
