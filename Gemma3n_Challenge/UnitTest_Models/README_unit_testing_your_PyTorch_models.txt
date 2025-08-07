After you have completed the Kaggle projects and have successfully generated your .PT files and downloaded from Kaggle, then you can test it before you use it against the server.py.

In this folder, there is the

test_custom_cnn_mobile.py
and
test_skin_cancer_model.py
and
test_health_check.py

a) Go here if you want to generate the .pt files for running the test_skin_cancer_model.py:
https://www.kaggle.com/code/prototypingai/ml-cnn-skin-malignant-benign/

b) Go here if you want to generate the .pt files for running the test_custom_cnn_mobile.py:
https://www.kaggle.com/code/prototypingai/ml-cnn-skin-cancer-classification/

c) Go here if you want to generate the .pt files for running the test_custom_cnn_mobile.py:
https://www.kaggle.com/code/prototypingai/ml-health-check


***** Don't use my file and path, put your own
Make sure you modify both .py files and put in your path to the image you want to test.

***** You can download the test images for Benign and Malignant from the datasets in the Kaggle projects.



cd to_your_working_directory
> Python .\test_custom_cnn_mobile.py
> Python .\test_skin_cancer_model.py
> Python .\test_health_check.py

This will give you a result that looks like this from both models:


from > Python .\test_skin_cancer_model.py
Predicted class: Melanocytic nevi
Predicted confidence: 0.998773992061615

from > Python .\test_custom_cnn_mobile.py
0 0.0010841587791219354
1 0.7248638868331909

from > Python .\test_health_check.py
Stroke risk prediction (probability): {prediction:.4f}
Risk Level:", "High" if prediction > 0.5 else "Low
You can update the user_inut json with test values.


Known Issues:
I'm still working on improving my models.  For this Hackathon, I'm not focused on accuracy.  I was working on generating a Pytorch machine learning model and outputting it to access an offline device such as my laptop and android device.  I focused on the integration and delivery.  I will return to improve the models.

If you submitted an image that yields terrible results then it may be that you need to ensure you use a good camera angle, lighting, and or image size.  For the time being, utilize the images in the project data set for testing.
I recommend testing using images from the Kaggle to get an understand of the classification first.





Disclaimer:
Production Readiness and Model Quality: The primary objective of this project is to demonstrate a viable technical pipeline for model conversion, specifically from a JIT-compiled PyTorch model to a TFLite format for integration with Android using AI Edge Gallery and other platforms like Ollama/Gemma3N on an offline laptop. This model is not production-ready. It has not been subjected to rigorous performance, accuracy, or bias testing. The code serves as a proof-of-concept for the conversion process, and any further development would require extensive model improvement and validation before real-world deployment.

Not a Medical Tool; No Medical Advice: The output from this model, whether used in an Android app or integrated with Ollama using an LLM like Gemma3N, is not a substitute for professional medical advice, diagnosis, or treatment. This model is an experimental tool for pattern recognition and should never be used as a replacement for consulting a qualified healthcare professional.







Feel free to modify those .py files and update a new image location.
