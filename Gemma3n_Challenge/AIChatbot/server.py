# This is the server.py file that handles talking to Ollama Gemma 3N model
# It also loads both the PyTorch models before sending it to Gemma 3N for summarization
# The two model files are needed in this same folder.
# 
# custom_cnn_mobile.pt (Benign or Malignant)
# skin_cancer_model.pt (Lesion type and cancer type)
#
import asyncio
import aiohttp
import datetime
import os
import json
import io
import torch
import numpy as np
import pandas as pd
import joblib
import re
 
import ollama
import base64
from PIL import Image
from torchvision import transforms
  

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.requests import Request

from typing import List

from langchain_community.chat_models import ChatOllama

from sklearn.preprocessing import StandardScaler




# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = torch.jit.load("custom_cnn_mobile.pt", map_location=device)
model.eval()

# Constants
MASTER_PROMPT = """
Answer the question base on the last question in the CONTENT xxxxxx {'role': 'user', 'content': xxxxxx}.
You can use the history of the conversation to understand the context as well as the history of the discussion; however, respond to the last question of the conversation.  Unless explicity asked by the user, you should only respond without json responses and avoid structured output. “Answer conversationally” “Do not include JSON, lists, or code blocks”
“Respond like a human speaking”


"""



MASTER_PROMPT_SKIN = """
You are a medical assistant, not a doctor. Given the following machine learning predictions about a skin image, generate a concise, empathetic summary. Do not introduce yourself or acknowledge the task. Do not say 'I understand' or 'Here's a summary'.

Be respectful, privacy-aware, and avoid making any diagnosis.

Your summary should:
• Interpret what the predictions may suggest.
• Include confidence percentages and what they may mean.
• Offer supportive next steps for the user.
• Remind the user this is not a diagnosis.

Be empathetic, concise, and informative.
Always include a disclaimer at least once in the chat but not repetitive if you see it in the history.
 

	"""

MASTER_PROMPT_HEALTH = """
You are a helpful medical assistant, but not a doctor. Use the following machine learning predictions to help interpret a health assessment based on the certain vital informations. Be cautious, respectful, and privacy-aware. Do not make a diagnosis.

Based on the model predictions above, provide a short, supportive summary that explains what these results *may* suggest, share any % of confidence and any meanings behind it, including what the user should do next. Remind the user that this is not a diagnosis and encourage a follow-up with a licensed medical professional. Keep it empathetic and concise.

Please include the exact texts:
 	**Thank you for submitting the assessment.** in the response.
 
	"""

MASTER_PROMPT_HEALTH_MORE_QUESTIONS = """
You are a helpful medical assistant, but not a doctor.  Be cautious, respectful, and privacy-aware. Do not make a diagnosis.

Help answer the user last question maybe use the model predictions info and historical conversations if it can relate to your response.
 
        """










### HEALTH ASSESSMENT
 

# Modify the validation for the "work type" question
# Validation for work type (example)
def validate_work_type_answer(answer: str):
    valid_responses = {
        "a": "Private", "private": "Private",
        "b": "Self Employed", "self employed": "Self Employed", "self-employed": "Self Employed",
        "c": "Children", "children": "Children",
        "d": "Government Job", "govt": "Government Job", "government": "Government Job",
        "e": "Never Worked", "never worked": "Never Worked"
    }
    
    answer_normalized = answer.strip().lower()
    return valid_responses.get(answer_normalized)


# Function to validate numeric responses (for BMI, age, glucose, etc.)
def validate_numeric_answer(answer: str):
    try:
        float(answer)
        return True
    except ValueError:
        return False


def validate_yes_no_answer(answer: str):
    return answer.strip().lower() in ["yes", "no"]

def validate_gender_answer(answer: str):
    return answer.strip().lower() in ["male", "female"]


def validate_work_type_answer(ans):
    normalized = ans.strip().lower()

    valid_values = {
        "a": "private",
        "b": "self employed",
        "c": "children",
        "d": "government",
        "e": "never worked",
        "private": "private",
        "self employed": "self employed",
        "self-employed": "self employed",
        "children": "children",
        "govt": "government",
        "government": "government",
        "never worked": "never worked"
    }

    return normalized in valid_values

 
# List of health-related questions with flexible answer validation
questions = [
    {
        "question": "Question #1) What is your work type? (A) Private  (B) Self Employed  (C) Children  (D) Government Job  (E) Never Worked",
        "answer_type": "categorical",
        "options": ["A", "B", "C", "D", "E", "Private", "Self Employed", "Self-employed", "Children", "Govt", "Government", "Never Worked"],
        "validate": validate_work_type_answer
        
    },
    {
        "question": "Question #2) What is your age?",
        "answer_type": "numeric",
        "validate": validate_numeric_answer
    },
    {
        "question": "Question #3) Do you have hypertension? (Yes/No)",
        "answer_type": "binary",
        "validate": validate_yes_no_answer
    },
    {
        "question": "Question #4) Are you married? (Yes/No)",
        "answer_type": "binary",
        "validate": validate_yes_no_answer
    },
    {
        "question": "Question #5) What is your gender? (Male/Female)",
        "answer_type": "binary",
        "validate": validate_gender_answer
    },
    {
        "question": "Question #6) Do you have heart disease? (Yes/No)",
        "answer_type": "binary",  # Changed from "boolean" to "binary"
        "options": ["yes", "no"],
        "validate": validate_yes_no_answer
    },
    {
        "question": "Question #7) Can you tell me your BMI?",
        "answer_type": "numeric",
        "validate": validate_numeric_answer  # Reused existing numeric validation
    },
    {
        "question": "Question #8) Do you smoke? (Yes/No)",
        "answer_type": "binary",  # Changed from "boolean" to "binary"
        "options": ["yes", "no"],
        "validate": validate_yes_no_answer
    },
    {
        "question": "Question #9) Can you tell me your average glucose level (mg/dL, usually between 70 and 200)?",
        "answer_type": "numeric",
        "validate": validate_numeric_answer  # Reused existing numeric validation
    },
    {
        "question": "Question #10) Do you live in an urban or rural area?",
        "answer_type": "categorical",
        "options": ["urban", "rural"],
        "validate": lambda answer: answer.lower() in ["urban", "rural"]
    }
]


def print_chat_history(chat_history):
    print("\n--- Chat History Start ---")
    for i, msg in enumerate(chat_history):
        print(f"Message {i}: role={msg.get('role')}, content={repr(msg.get('content'))}")
    print("--- Chat History End ---\n")
 

# Function to handle the Q&A flow and validation
def get_last_qa_pair(chat_history):
    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        if msg.get("role") == "assistant":
            content = msg["content"]

            # Try parsing JSON only if you expect JSON object, else fallback to string
            try:
                q_data = json.loads(content)
                # If it's a dict, we have old-style JSON
                if isinstance(q_data, dict):
                    question_text = q_data.get("question", "")
                else:
                    # It's a string (your new style)
                    question_text = q_data
            except Exception:
                # If content is plain string not JSON, just use as is
                question_text = content

            if i + 1 < len(chat_history):
                next_msg = chat_history[i + 1]
                if next_msg.get("role") == "user":
                    user_answer = next_msg.get("content", "").strip()
                    return question_text, user_answer

    return None, None


 

def get_question_index_from_text(question_text):
    match = re.search(r"Question #(\d+)", question_text)
    if match:
        return int(match.group(1)) - 1
    return None

def get_last_answered_question_index(chat_history):
    q_text, user_answer = get_last_qa_pair(chat_history)

    if q_text is None or user_answer is None:
        return 0  # start from first question

    q_index = get_question_index_from_text(q_text)
    if q_index is None:
        return 0

    validator = questions[q_index]["validate"]
    if validator(user_answer):
        return q_index + 1  # next question
    else:
        return q_index  # repeat current question




 
# Function to track the progress using the chat historydef get_last_answered_question_index(chat_history):
 


 
# Health assessment flow
def build_health_assessment(chat_history):
    user_input = {}

    def normalize(answer):
        return answer.strip().lower()

    # Maps
    gender_map = {"male": 0, "female": 1}
    married_map = {"yes": 1, "no": 0}
    binary_map = {"yes": 1, "no": 0}
    residence_map = {"rural": 0, "urban": 1}
    smoking_map = {"yes": 1, "no": 0}
    work_type_map = {
        "private": 0,
        "self employed": 1,
        "self-employed": 1,
        "children": 2,
        "government": 3,
        "govt": 3,
        "never worked": 4
    }

    # Scan chat history for each Q&A pair
    for i in range(len(chat_history) - 1):
        msg = chat_history[i]
        next_msg = chat_history[i + 1]

        if msg.get("role") != "assistant" or next_msg.get("role") != "user":
            continue

        q_text = msg.get("content", "")
        a_text = normalize(next_msg.get("content", ""))

        # Identify question number
        match = re.search(r"Question #(\d+)", q_text)
        if not match:
            continue

        q_num = int(match.group(1))

        # Map each question to the correct field
        if q_num == 1:
            user_input["work_type"] = work_type_map.get(a_text, 0)
        elif q_num == 2:
            try:
                user_input["age"] = int(float(a_text))
            except:
                pass
        elif q_num == 3:
            user_input["hypertension"] = binary_map.get(a_text, 0)
        elif q_num == 4:
            user_input["ever_married"] = married_map.get(a_text, 0)
        elif q_num == 5:
            user_input["gender"] = gender_map.get(a_text, 0)
        elif q_num == 6:
            user_input["heart_disease"] = binary_map.get(a_text, 0)
        elif q_num == 7:
            try:
                user_input["bmi"] = float(a_text)
            except:
                pass
        elif q_num == 8:
            user_input["smoking_status"] = smoking_map.get(a_text, 0)
        elif q_num == 9:
            try:
                user_input["avg_glucose_level"] = float(a_text)
            except:
                pass
        elif q_num == 10:
            user_input["Residence_type"] = residence_map.get(a_text, 0)

    print("\n\n\nUser_Input:\n", user_input)
    print("\n\n\nUser_Input:\n", json.dumps(user_input, indent=4))
    
    return user_input


async def health_assessment_flow(chat_history):
    #print_chat_history(chat_history)

    current_question_index = get_last_answered_question_index(chat_history)
    #print("\ncurrent_question_index", current_question_index)

    if current_question_index >= len(questions):
        # All answered — build final user input
        user_input = build_health_assessment(chat_history)
        #print("User input collected:", user_input)

        risk_prediction = None
        # Predict stroke risk
        try:
            probability, risk_level = predict_stroke_risk(user_input)
            risk_prediction =  f"✅ Assessment Complete!\n" + f"Stroke Risk Prediction: **{probability:.4f}**\n" + f"Risk Level: **{risk_level}**"
             
        except Exception as e:
            return f"⚠️ Error in prediction: {e}"

        # Call Ollama/Gemma3N
        try:
             prompt = f"{MASTER_PROMPT_HEALTH} - Predictions from Machine Learning Model Assessment Completed:: {risk_prediction} User Chat History: {chat_history}" 	
             return await call_ollama_model(text=prompt)

        except Exception as e:
            return f"⚠️ Error in prediction: {e}"


    # Otherwise, continue asking next question
    question = questions[current_question_index]
    return question["question"]

 

  

 


 
#####
## HEALTH CHECK CNN
#####
# Define model again for loading

class CardioNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load model and scaler (only once)
#scaler = joblib.load("scaler.save")

input_dim = 10
modelHealth = CardioNet(input_dim=input_dim)
modelHealth.load_state_dict(torch.load("cardio_model.pt"))
modelHealth.eval()


def predict_stroke_risk(user_input: dict) -> tuple:
    """
    Predict stroke risk from user_input.
    Returns: (probability: float, risk_level: str)
    """

    # Define the exact order of features used during model training
    feature_order = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ]

    try:
        # Load the saved scaler
        scaler = joblib.load("scaler.save")

        # Reorder the input dict according to expected column order
        input_df = pd.DataFrame([[user_input[feature] for feature in feature_order]], columns=feature_order)

        input_scaled = scaler.transform(input_df)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = modelHealth(input_tensor).item()

        risk_level = "High" if prediction > 0.5 else "Low"
        return prediction, risk_level

    except Exception as e:
        raise RuntimeError(f"Error in prediction: {e}")

 


#####################













# Define the transform once globally
status_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
classification_transform = transforms.Compose([
    transforms.Resize((75, 100)),  # match training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

status_model_cnn = torch.jit.load("custom_cnn_mobile.pt", map_location=device)
status_model_cnn.eval()
status_model_cnn.to(device)


# Class labels (must match training)
class_names = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis', 'Basal cell carcinoma',
               'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']


# Define your model architecture (same as training)
class SkinCNN(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.4)
        )

        # Make output shape fixed, regardless of input image size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((6, 6))  # Output will always be 64 x 6 x 6 = 2304

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 6 * 6, 128),  # <- fixed size
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


# Initialize model and load state dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model_cnn = torch.jit.load("skin_cancer_model.pt", map_location=device)
classification_model_cnn.eval()
classification_model_cnn.to(device)
 
 

# Define the same model architecture used during training
class CardioNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



 



def predict_image(model, image_path=None, image=None):
    if image_path:
        image = Image.open(image_path).convert("RGB")
    elif image is None:
        raise ValueError("No image or image_path provided.")

    input_tensor = classification_transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        output = model(input_tensor)
        # Assuming classification, get predicted class index
        _, predicted = torch.max(output, 1)
        return predicted.item()  # or process output further as needed

 

async def homepage(request: Request):

    try:

        payload = await request.body()

        string = payload.decode("utf-8")

        response_q = asyncio.Queue()

 

        await request.app.model_queue.put((string, response_q))

       

        # Adding a timeout for getting the output

        try:

            output = await asyncio.wait_for(response_q.get(), timeout=60.0)  # 30 seconds timeout

        except asyncio.TimeoutError:

            return JSONResponse({"error": "Request timed out."}, status_code=504)

 

        return JSONResponse(output)

 

    except Exception as e:

        print(f"Error in homepage: {e}")

        return JSONResponse({"error": "An error occurred while processing your request on the Reverse Proxy."}, status_code=500)

 

  


################################
 
 

# Prediction function
def predict_image(model, image, threshold=0.5):
    input_tensor = status_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)  # (1,1)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > threshold else 0
        return prediction, prob

# Main handler
def predict_base64_image(base64_str, model, threshold=None):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = classification_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()

        return {
            "class_index": predicted_class,
            "class_name": class_names[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        return {
            "error": str(e)
        }

 

async def handle_text_and_image(text, image_base64, model_name):

    skin_class = ""
    skin_status = ""


    if image_base64 and any(k in text.lower() for k in ["skin", "cancer", "mole"]):
        try:


            result = predict_base64_image(image_base64, classification_model_cnn)
            #print("\n\nPrediction result:", result)


            if "error" in result:
               print(f"❌ Error during image prediction: {result['error']}")
            skin_class =  (f"🧪 Skin Check Result: **{result['class_name']}** " + f"(Confidence: {result['confidence']:.2f})")
            #print("\n", skin_class)#

        except Exception as e:
            print(f"❌ Error during skin_class image prediction: {str(e)}")


    if image_base64 and any(k in text.lower() for k in ["skin", "cancer", "mole"]):
        try:
            image_data = base64.b64decode(image_base64)
            image_obj = Image.open(io.BytesIO(image_data)).convert("RGB")

            prediction, prob = predict_image(model=status_model_cnn, image=image_obj)
            class_namesX = ["Benign", "Malignant"]


	    # second model
	    

            skin_status = f"🧪 Skin Check Result: **{class_namesX[prediction]}** (Confidence: {prob:.2f})"

        except Exception as e:
            print( f"❌ Error during skin_status image prediction: {str(e)}")

    if image_base64 and any(k in text.lower() for k in ["skin", "cancer", "mole"]):


        prompt = f"{MASTER_PROMPT_SKIN} Here is the skin classification: {str(skin_class)} and here is the skin status: {str(skin_status)}. User Question: {text}"
    else:
        prompt = text
   
    #print(prompt)
    #return f"{str(skin_status)}   {str(skin_class)}"
    # If not a skin/mole/cancer related query, forward to Ollama
    return await call_ollama_model(text=prompt, image_base64=image_base64, model=model_name)

  
async def call_ollama_model(text: str, image_base64: str = None, model: str = "gemma3:4b") -> str:
    try:
        client = ollama.Client()

        kwargs = {
            "model": model,
            "prompt": text,
        }

        print("\n\n", text)
        #print("\n", model)
        #print("\n", image_base64)

        if image_base64:          
            kwargs["images"] = [image_base64]  # list of base64 strings

        response = client.generate(**kwargs)

        return response.get("response", "[No response from Ollama]").strip()

    except Exception as e:
        return f"❌ Error calling Ollama: {str(e)}"


#### SERVER LOOP ####
# Main server loop to handle health assessment and model integration
async def server_loop(q):
    """
    Background worker that processes requests from the queue.
    """
    while True:
        try:
            string, response_q = await q.get()
            image = None
            try:
                # Parse input string to JSON
                if isinstance(string, str):
                    parsed = json.loads(string)
                else:
                    parsed = string

                # Ensure parsed is a dict before accessing keys
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON object but got {type(parsed)}")

                model = parsed.get("model", "gemma3:4b").strip()
                messages = parsed.get("messages", [])
                if not messages or "content" not in messages[0]:
                    raise ValueError("Payload missing 'messages' or 'content'")

                first_msg = messages[0]
                content_raw = first_msg["content"]

                # Parse content if it's a JSON string list/dict, else keep as is
                try:
                    content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
                except json.JSONDecodeError:
                    content = content_raw  # keep raw if not valid JSON

                text_parts = []
                image = None
 

                if isinstance(content, list):

                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "image":
                            image = part.get("image")
                        else:
                            # Fallback: safe stringify any dict or non-str item
                            if isinstance(part, dict):
                                text_parts.append(str(part.get("text") or part))
                            else:
                                text_parts.append(str(part))

                            text_parts = [t for t in text_parts if t.strip() != MASTER_PROMPT.strip()]
                            text_parts.append(MASTER_PROMPT.strip())
                elif isinstance(content, dict):

                    # single dict content (text or image)
                    if content.get("type") == "text":
                        text_parts.append(content.get("text", ""))
                    elif content.get("type") == "image":
                        image = content.get("image")
                else:
 
                    # content is just plain string or other
                    text_parts.append(str(content))

                text = " ".join(text_parts).strip()


            except Exception as e:
                error_msg = f"Invalid JSON input or missing fields: {e}"
                print(error_msg)
                await response_q.put(error_msg)
                continue

            print(f"\n🕐 [{datetime.datetime.now()}] Calling Ollama with model='{model}'")
            
            if image:
                print("Image: [base64 image received]")

            ## uncomment this to see the debug text in the command prompt that you ran python .\proxy.py
            #print("\n\n\n[DEBUG]:", text.lower())

            if "health" in text.lower() and "Thank you for submitting the assessment".lower() in text.lower():
                text = f"{MASTER_PROMPT_HEALTH_MORE_QUESTIONS} {text}"

            if "health" in text.lower() and "Thank you for submitting the assessment".lower() not in text.lower():
                print("\nHEALTH ASSESSMENT ENTERED")
                # Pass full chat history list to health_assessment_flow to track progress
                result = await health_assessment_flow(content if isinstance(content, list) else [content])

                result = json.dumps(result)

            else:
                # Handle other model requests (e.g., skin cancer, etc.)
                result = await handle_text_and_image(text, image, model_name=model)

             

            await response_q.put(result)

        except Exception as e:
            print(f"❌ Fatal error in server loop: {e}")
            break




 



 
  
 

################################





 

def startup():
    

 #   try:
 #       load_dotenv()
 #   except Exception as e:
 #       print(f"Error loading .env file: {e}")


    q = asyncio.Queue()

    app.model_queue = q


    print("\nStarting Async Task Web app")
    asyncio.create_task(server_loop(q))

 

app = Starlette(

    routes=[

        Route("/", homepage, methods=["POST"]),

    ],

    on_startup=[startup]

)

