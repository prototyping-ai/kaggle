from PIL import Image
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Example transform (must match training)
transform = transforms.Compose([
    transforms.Resize((75, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load and preprocess your test image
image_path = 'c:/users/xxxxxxxxxxxxxxxxx/downloads/B1010.jpg'
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device

# Run inference
class_cnn_model = torch.jit.load("skin_cancer_model.pt", map_location=device)
class_cnn_model.to(device)
class_cnn_model.eval()
with torch.no_grad():
    outputs = class_cnn_model(input_tensor)
    probs = torch.softmax(outputs, dim=1)  # shape: (1, num_classes)
    predicted_index = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_index].item() 

# Map to class name
class_names = [
    'Melanocytic nevi', 'Melanoma', 'Benign keratosis', 'Basal cell carcinoma',
    'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma'
]
predicted_label = class_names[predicted_index]
print(f"Predicted class: {predicted_label}")
print(f"Predicted confidence: {confidence}")