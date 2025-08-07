######################
READ ME / INSTALLATION STEPS
######################
This instruction is for Phase 1 of this project which allows you to run a local version of the aiChat integrated with Ollama pulling Gemma 3N 4B to test the two machine learning models.

This repository is for the Kaggle project:
https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/local-ollama-gemma-3n-4b-llm-with-skin-cancer-dete

1) create a folder on your laptop.

2) Download all of these files to that folder: (aiChat.html, proxy.py, server.py, custom_cnn_model.pt, skin_cancer_model.pt, scaler.save, scaler_health_stroke.save, cardio_model.pt)

3) Install Ollama ( ollama.com/download )

4) Pull the latest Gemma 3N 4B model 
	> ollama run gemma3:4B 
or 
	> ollama pull gemma3:4B
start the ollama server using
	> ollama serve

5) Install Python ( www.python.org/downloads/ )

6) You will need to run pip install on all of the packages needed. Take a look at my two Kaggle projects to see the pip list and also notes above.

pip install ollama fastapi uvicorn starlette pandas joblib
pip install torchvision transforms
pip install numpy asyncio aiohttp
pip install langchain_community
pip install scikit-learn

To see a full list of all my pip list goto this website:
https://www.kaggle.com/code/prototypingai/ml-cnn-skin-cancer-classification/edit/run/253692408#Summary-of-What-I-Learned


7) Run the proxy.py which takes care of the CORS issues where your local aiChat.html is loading a local port 80 communication to the Python Machine Learning models and directing traffic to Ollama and Gemma 3N. (e.g. 

In a command prompt type these two commands (assuming you have Python already in Path):
	> cd your_working_directory 

	> Python .\proxy.py )

8) Open the aiChat.html file in any browser

9) Start chatting with the Ollama/Gemma 3N

10) If you mention anything related to skin, cancer or mole then please submit an image. 
If you need images to test then use the images in my Kaggle projects.
You can use images in my two kaggle projects:

https://www.kaggle.com/code/prototypingai/ml-cnn-skin-cancer-classification

https://www.kaggle.com/code/prototypingai/ml-cnn-skin-malignant-benign/

11) You can look at my video or images posted on my Kaggle Gemma 3N projects:
 
12) To improve the models and create a new .pt file then use the two Kaggle projects to improve accuracy.


Content:

a) aiChat.html
* This is the html file containing the html, javascript and css that will work on any browser. It was tested on both Chrome and Microsoft Edge on a Windows 11 Laptop. It should also work on a Mac.  The javascript calls the localhost:80 port 80 which talks to the reverse proxy.

b) proxy.py
* This is the reverse proxy that handles any CORS issues. Basically your aiChat.html file can call the port 80 web server without any issues.  the sendMessage calls http://localhost/ as the rest api call.  
This proxy.py handles the starlette app that runs the server.py file

c) server.py
* This file is basically the Python code that talks to Ollama Gemma 3N B as well as two of the machine learning models.  Put the proxy.py, server.py and the model files into the same folder.

d) custom_cnn_mobile.pt
* This is the machine learning model against my Kaggle project that determines if a skin lesion is benign or malignant.  It takes in a base 64 encoded image from the aiChat.html.  It responds with an accuracy %.
Use the link below to improve accuracy and deploy a new .pt file in the same folder as proxy.py and server.py.

https://www.kaggle.com/code/prototypingai/ml-cnn-skin-malignant-benign/

e) skin_cancer_model.pt
* This is the other machine learning model against my Kaggle project that looks at a skin lesion and classifies the type of the skin.  It takes a base 64 encoded image from the aiChat.html. It responds with the classification type and accuracy %.  Use the link below to improve accuracy and deploy a new .pt file in the same folder as proxy.py and server.py.

https://www.kaggle.com/code/prototypingai/ml-cnn-skin-cancer-classification

f) scaler.save
* This is needed for the custom_cnn_mobile.pt which saves information for the sckit-learn for persistent and transform images.

g) cardio_model.pt
* This is the health stroke risk evaluation.  This model takes in user information like their BMI, glucose, smoking status, and other private information locally and sends to this model for classifying a user having a high or low risk of stroke.

h) scaler_health_stroke.save
* This is needed for the custom_cnn_mobile.pt which saves information for the sckit-learn for persistent and transform images.


######################
Issues:
######################
    a) The port for the reverse proxy is on port 80 by default so make sure you don't have port 80 utilized for anything else.  If you need to change to a different port then modify the proxy.py file and on the uvicorn.run line change the port to whatever you want.
e.g.

uvicorn.run("server:app", host="0.0.0.0", port=XYZ)

You also need to modify the aiChat.html and update that port as well in this code snippet at the callOllamaAPI function

 // code 
 try {
    const response = await fetch('http://localhost:XYZ/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });



    b) After click on Send on the chatbot, you are getting errors in the browser in the console log.  Ensure you start the Ollama server and also pull the Gemma3n:4B model.
    c) The chatbot is not responding with any results from the ML models.  Make sure you deploy the .pt files and the scaler.save in the same folder as the server.py file.









######################
 Disclaimer
######################
Production Readiness and Model Quality: The primary objective of this project is to demonstrate a viable technical pipeline for model conversion, specifically from a JIT-compiled PyTorch model to a TFLite format for integration with Android using AI Edge Gallery and other platforms like Ollama/Gemma3N on an offline laptop. This model is not production-ready. It has not been subjected to rigorous performance, accuracy, or bias testing. The code serves as a proof-of-concept for the conversion process, and any further development would require extensive model improvement and validation before real-world deployment.

Not a Medical Tool; No Medical Advice: The output from this model, whether used in an Android app or integrated with Ollama using an LLM like Gemma3N, is not a substitute for professional medical advice, diagnosis, or treatment. This model is an experimental tool for pattern recognition and should never be used as a replacement for consulting a qualified healthcare professional.


######################
 Credits
######################
Health Models:
https://repositories.lib.utexas.edu/items/7ce65ba2-df98-4ab5-8887-a02c04a2a1fc
Datasets:
https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
https://www.kaggle.com/datasets/murtozalikhon/skin-cancer-classification