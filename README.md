Video of the app working locally on a laptop with Ollama and Gemma 3N 4B pulled.  Python is needed:
https://www.youtube.com/watch?v=Mr7RlW\_o7Co

This github is for the Gemma 3N Kaggle Challenge:

https://www.kaggle.com/competitions/google-gemma-3n-hackathon/writeups/local-ollama-gemma-3n-4b-llm-with-skin-cancer-dete

Machine Learning models:

https://www.kaggle.com/code/prototypingai/ml-health-check/edit

https://www.kaggle.com/code/prototypingai/ml-cnn-skin-cancer-classification

https://www.kaggle.com/code/prototypingai/ml-cnn-skin-malignant-benign

This is a front-end Chatbot that works locally on a laptop using Ollama and Python running Starlette app.

This project will show case the capabilities to integrate an offline device access to Ollama pulling Gemma 3N 4B as the LLM to help answer questions against the ChatBot interface.

This project integrates a Chatbot interface (aiChat.html) against 3 Machine Learning models to provide these capabilities:

a) Check to see if a skin lesion or mole is benign or malignant

b) Check to see what type of skin cancer does the image classify as

c) It also has a health questionnaire that sends to the machine learning model to determine if a person's health has a high or low stroke risk.

d) It has an Ask Image functionality to allow your Chatbot to accept text and image to have Gemma 3N explain/describe the image.

e) I also included several offline games that communicates with the offline Gemma 3N LLM



The code could also be deployed to a webserver.  The machine learning models could also be deployed to AI Edge Gallery using the .tflite files.

Thank you,

Jack Dinh



