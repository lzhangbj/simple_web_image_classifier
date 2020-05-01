import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

import gunicorn
print(gunicorn.__version__)

imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = torch.load('densenet.pt')
# model = torch.load("densenet121-a639ec97.pth")
model.eval()

# resnet = models.resnet18(pretrained=True)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]