import torch
import base64
import io
from PIL import Image
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = torch.jit.load("custom_cnn_mobile.pt", map_location=device)
cnn_model.eval()

# Constants
img_size = 112
class_namesX = ["Benign", "Malignant"]

# Transform
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

 


# Prediction function
def predict_image(model, image, threshold=0.5):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)  # (1, 1)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > threshold else 0
        return prediction, prob

# Main test function
def test_model_with_base64(image_base64, text=""):
    if image_base64 and any(k in text.lower() for k in ["skin", "cancer", "mole"]):
        try:
            image_data = base64.b64decode(image_base64)
            image_obj = Image.open(io.BytesIO(image_data)).convert("RGB")

            prediction, prob = predict_image(model=cnn_model, image=image_obj)
            skin_status = f"🧪 Skin Check Result: **{class_namesX[prediction]}** (Confidence: {prob:.2f})"
            print(skin_status)
        except Exception as e:
            print(f"❌ Error during skin_status image prediction: {str(e)}")
    else:
        print("⚠️ No valid image or keyword provided.")

# Example usage
if __name__ == "__main__":
    # You must replace this with your actual base64 string
    image_path = 'c:/users/xxxxxxxxxxxxxxxxxxx/downloads/B1010.jpg'
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    #input_tensor = transform(image).unsqueeze(0).to(device)
    threshold=0.5
    with torch.no_grad():
        output = cnn_model(input_tensor)  # (1, 1)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > threshold else 0
    print(prediction, prob)

  
    # You must replace this with your actual base64 string
    image_path = 'c:/users/xxxxxxxxxxxxxxxxx/downloads/M1014.jpg'
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    #input_tensor = transform(image).unsqueeze(0).to(device)
    threshold=0.5
    with torch.no_grad():
        output = cnn_model(input_tensor)  # (1, 1)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > threshold else 0
    print(prediction, prob)


