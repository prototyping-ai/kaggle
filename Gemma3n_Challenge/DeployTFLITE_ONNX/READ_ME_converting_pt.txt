In this folder, there are three Python codes that you can run locally on your laptop or on Kaggle.

1.0) 

After you go to the Kaggle projects and create a Pytorch .py file then you can run this to generate the tflite file.
https://www.kaggle.com/code/prototypingai/ml-cnn-skin-malignant-benign/
you will need to obtain the custom_cnn_mobile.pt as an output save in your to_working_directory

1a) 

Then you want to run this .py file so that it generates the tflite
tflite_cnn_custom_model.py 

This will take in the .pt file that you generated in Kaggle.   
> cd to_working_directory
> Python .\tflite_cnn_custom_model.py

If you run the Python codes, you will generate these files:
custom_cnn_mobile_model.tflite
custom_cnn_mobile_model.onnx


1b)
After you go to the Kaggle projects and create a Pytorch .py file then you can run this to generate the tflite file.

https://www.kaggle.com/code/prototypingai/ml-cnn-skin-cancer-classification/

Then you want to run this .py file so that it generates the tflite
skin_cancer_model.pt

Then you want to run this .py file so that it generates the tflite
tflite_skin_classification.py

This will take in the skin_cancer_model.pt file that is generated from Kaggle which you will need to download.
> cd to_working_directory
> Python .\tflite_skin_classification.py
If you run the Python codes, you will generate these files:

skin_cancer_model.tflite
skin_cancer_model.onnx


1c)
After you go to the Kaggle projects and create a Pytorch .py file then you can run this to generate the tflite file.

https://www.kaggle.com/code/prototypingai/ml-health-check

Then you want to run this .py file so that it generates the tflite
cardio_model.pt

Then you want to run this .py file so that it generates the tflite
tflite_health_stroke_eval.py

This will take in the cardio_model.pt file that is generated from Kaggle which you will need to download.
> cd to_working_directory
> Python .\tflite_health_stroke_eval.py
If you run the Python codes, you will generate these files:

cardio_stroke_model.tflite
cardio_stoke_model.onnx



2.0 - For information purposes on integrating with AI Edge Gallery (Not fully completed in Phase 2):

These two Python codes will generate the files you need suitable for AI Edge Gallery.
This is my Phase 2 of the project for learning purposes in AI Edge Gallery but not for the Hackathon due to time constraints.


If you are using AI Edge Gallery, you can call these custom models in the
LlmChatViewModel.kt file that I started working.  It is not complete but it is a good start.
You can put those .tflite files in your assets folder.







3.0) Installation of Python classes needed to run those Python codes to generate .tflite files:

You may need to install some Python classes needed.

start with:
pip install onnx onnx-tf onnxruntime keras



# My laptop
pip list

Package Version

absl-py 2.3.1

aiohappyeyeballs 2.6.1

aiohttp 3.12.15

aiosignal 1.4.0

annotated-types 0.7.0

anyio 4.9.0

astunparse 1.6.3

attrs 25.3.0

certifi 2025.7.14

charset-normalizer 3.4.2

click 8.2.1

colorama 0.4.6

coloredlogs 15.0.1

dataclasses-json 0.6.7

fastapi 0.116.1

filelock 3.18.0

flatbuffers 25.2.10

frozenlist 1.7.0

fsspec 2025.7.0

gast 0.6.0

google-pasta 0.2.0

greenlet 3.2.3

grpcio 1.74.0

h11 0.16.0

h5py 3.14.0

httpcore 1.0.9

httpx 0.28.1

httpx-sse 0.4.1

humanfriendly 10.0

idna 3.10

intel-cmplr-lic-rt 2025.2.0

intel-opencl-rt 2025.2.0

Jinja2 3.1.6

joblib 1.5.1

jsonpatch 1.33

jsonpointer 3.0.0

keras 3.11.1

langchain 0.3.27

langchain-community 0.3.27

langchain-core 0.3.72

langchain-text-splitters 0.3.9

langsmith 0.4.8

libclang 18.1.1

Markdown 3.8.2

markdown-it-py 3.0.0

MarkupSafe 3.0.2

marshmallow 3.26.1

mdurl 0.1.2

ml_dtypes 0.5.3

mpmath 1.3.0

multidict 6.6.3

mypy_extensions 1.1.0

namex 0.1.0

networkx 3.5

numpy 2.3.2

ollama 0.5.1

onnx 1.18.0

onnx-tf 1.6.0

onnxruntime 1.22.1

opt_einsum 3.4.0

optree 0.17.0

orjson 3.11.1

packaging 25.0

pandas 2.3.1

pillow 11.3.0

pip 25.1.1

propcache 0.3.2

protobuf 6.31.1

pydantic 2.11.7

pydantic_core 2.33.2

pydantic-settings 2.10.1

Pygments 2.19.2

pyreadline3 3.5.4

python-dateutil 2.9.0.post0

python-dotenv 1.1.1

pytz 2025.2

PyYAML 6.0.2

requests 2.32.4

requests-toolbelt 1.0.0

rich 14.1.0

scikit-learn 1.7.1

scipy 1.16.1

setuptools 80.9.0

six 1.17.0

sniffio 1.3.1

SQLAlchemy 2.0.42

starlette 0.47.2

sympy 1.14.0

tbb 2022.2.0

tcmlib 1.4.0

tenacity 9.1.2

tensorboard 2.20.0

tensorboard-data-server 0.7.2

tensorflow 2.20.0rc0

termcolor 3.1.0

threadpoolctl 3.6.0

torch 2.7.1

torchvision 0.22.1

typing_extensions 4.14.1

typing-inspect 0.9.0

typing-inspection 0.4.1

tzdata 2025.2

urllib3 2.5.0

uvicorn 0.35.0

Werkzeug 3.1.3

wheel 0.45.1

wrapt 1.17.2

yarl 1.20.1

zstandard 0.23.0


