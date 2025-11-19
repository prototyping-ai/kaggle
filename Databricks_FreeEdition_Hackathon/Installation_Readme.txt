This installation instruction is to download and install the Databricks Free Edition - Hackathon project that I demoed.  

1) Install Ollama using the OllamaSetup.exe in the Software_Downloads folder or you can download from Ollama website. https://ollama.com/

2) Once you have installed then run this ollama command to pull the Gemma3:4b module
ollama pull Gemma3:4b

3) Update the server.py file and update the URL to your Databricks model endpoint and update the Databricks token:

DATABRICKS_BASE_URL = "https://dbc-.cloud.databricks.com"
DATABRICKS_TOKEN = ""
DATABRICKS_ENDPOINT = "quickstart-agent-endpoint"

Obtain the information from above using your own instance information.  My token will expire so please create your own token in your Databricks -> profile -> developer access

4) Import the two databricks notebook and run them

a) Build your first AI agent is the model serving "quickstart-agent-endpoint".  In this agent file, all I did differently from the original out-of-the-box is to remove the Python exec function that will not work when calling outside of the Databricks instance. However, once you run and register the model then it will create an end-point in Databricks using the serverless compute.

b) Import and run the Health_Model.dbc, this will create for you the files in the Model_Output folder: cardio_modelX.pt and scaler_health_strokeX.save, these two are called in the server.py file.

5) Then on your command prompt on a laptop/desktop, CD to the Front_End folder and run Python.
python .\proxy.py

6) Then open up a browser to load the aiChat.html file which is your interface.

I will update this document regularly to ensure everything works.  Any files that I include with a most recent date will be a cleaner version; however, the original files with the time stamp for the hackathon will be preserved.


Link to my demo on YouTube:
https://www.youtube.com/watch?v=343OzAOVnNY


Link to my Linked In:
https://www.linkedin.com/in/jack-dean-9838b76b/


Reddit Post:
https://www.reddit.com/r/databricks/comments/1p0mtr3/databricks_free_edition_amazing_projects/

 

