# This file is for a reverse proxy so that your offline aiChat.html file can interact with 
#   a webpage that has CORS configured under FastAPI.
#   It then calls the server.py file which is the Python that calls the .pt models from Kaggle.
#   It also calls Ollama and Gemma 3N 4B
#
from pip._internal import main as pipmain
#pipmain(['install', 'uvicorn']) 
#pipmain(['install', 'starlette']) 

# Databricks notebook source
import subprocess

# Run the pip list command and capture the output
proc = subprocess.run(['pip', 'list'], stdout=subprocess.PIPE, text=True)

# Print the output of the pip list command
print(proc.stdout)


    
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server import app  # Import the FastAPI instance from the server module

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# No need to define additional routes here since they are in server.py

if __name__ == "__main__":
    import uvicorn
    import getpass

# Get the username of the currently logged-in user
    user_name = getpass.getuser()

    print(f"The current user logged in is: {user_name}")
#    uvicorn.run("server:app", host="0.0.0.0", port=7780)    
    uvicorn.run("server:app", host="0.0.0.0", port=80)    
