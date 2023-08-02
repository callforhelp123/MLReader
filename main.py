import numpy as np
import zipfile
import pandas as pd
import os
import tensorflow as tf
import tensorflow.lite as tflite

from tflite_support import metadata
from uvicorn import run
from fastapi import UploadFile, File, FastAPI

#----------------------------------------------------------------------------------------------------------------------#
# The following section houses the machine learning model methods for validation, and evaluation.                      #
# There is one set of methods for network hosted images (with an accessible URL) that works via image paths.           #
# There is another set of methods for locally hosted images that works by uploading the image file to the server.      #
#----------------------------------------------------------------------------------------------------------------------#

def load_labels(filename):
    """This function opens the dictionary file and returns the in-order items for the ML model"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def unzip_model(ml_model_path, path_to_unzip_to=os.getcwd()):
    """This function is used to unzip the model file to extract the appropriate dictionary"""
    with zipfile.ZipFile(ml_model_path, 'r') as zip_ref:
        zip_ref.extractall(path_to_unzip_to)

def prepare_interpreter(ml_model):
    """This function prepares the ML interpreter given the model path"""
    metadata_displayer = metadata.MetadataDisplayer.with_model_file(ml_model)
    list_of_files = metadata_displayer.get_packed_associated_file_list()
    dict_file = list_of_files[0]  # This is the name of the dictionary file (usually dict.txt)
    unzip_model(ml_model)  # This unzips the model file (tflite) to extract the associated metadata

    interpreter = tflite.Interpreter(ml_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    labels = load_labels(dict_file)
    return interpreter, input_details, output_details, width, height, labels

def analyze_image(ml_model_path, image_path, positive_threshold):
    """This function is used to call the interpreter and classifiers given the model, image path and thresholds"""
    print(type(image_path))
    interpreter, input_details, output_details, width, height, labels = prepare_interpreter(ml_model_path)
    interpreter_details = [interpreter, input_details, output_details, width, height, labels]

    return classify_image(interpreter_details, image_path, positive_threshold)

def classify_image(interpreter_details, image_path, positive_threshold=0.75):
    """ This function is used to classify a network image with a given threshold.
    If the image has a value above the threshold, then classify as positive, else negative"""
    interpreter, input_details, output_details, width, height, labels = interpreter_details
    img = np.uint8(tf.image.resize(tf.io.decode_image(tf.io.read_file(image_path)), [width, height], method=tf.image.ResizeMethod.BILINEAR))
    # if the image has an extra channel, reduce to 3 channels
    if img.shape[2] == 4:
        img = img[:, :, :3]
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    index_of_positive = [i for i, s in enumerate(labels) if 'Positive' in s]
    prob = float(results[index_of_positive] / 255.0)
    if prob >= positive_threshold:
        text_result = 'Positive'
    else:
        text_result = 'Negative'
    result_dict = {'Result': text_result, 'Positive Probability': prob}
    return result_dict

def analyze_image_local(ml_model_path, image, positive_threshold):
    """This function is used to call the interpreter and classifiers given the model, image path and thresholds"""
    print(type(image))
    interpreter, input_details, output_details, width, height, labels = prepare_interpreter(ml_model_path)
    interpreter_details = [interpreter, input_details, output_details, width, height, labels]
    return classify_image_local(interpreter_details, image, positive_threshold)

def classify_image_local(interpreter_details, image, positive_threshold=0.75):
    """ This function is used to classify a local image with a given threshold
    If the image has a value above the threshold, then classify as positive, else negative"""
    interpreter, input_details, output_details, width, height, labels = interpreter_details
    img = np.uint8(tf.image.resize(tf.io.decode_image(image), [width, height], method=tf.image.ResizeMethod.BILINEAR))
    # if the image has an extra channel, reduce to 3 channels
    if img.shape[2] == 4:
        img = img[:, :, :3]
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    index_of_positive = [i for i, s in enumerate(labels) if 'Positive' in s]
    prob = float(results[index_of_positive] / 255.0)
    if prob >= positive_threshold:
        text_result = 'Positive'
    else:
        text_result = 'Negative'
    result_dict = {'Result': text_result, 'Positive Probability': prob}
    return result_dict

#------------------------------------------------------------------------------------------------------------

def validate_image(ml_model_path, image, positive_threshold):
    """This function is used to call the interpreter and classifiers given the model, image path and thresholds"""
    interpreter, input_details, output_details, width, height, labels = prepare_interpreter(ml_model_path)
    interpreter_details = [interpreter, input_details, output_details, width, height, labels]
    return validate_image_local(interpreter_details, image, positive_threshold)

def validate_image_local(interpreter_details, image, invalid_threshold=0.35):
    """ This function is used to classify a local image with a given threshold
    If the image has a value above the threshold, then classify as positive, else negative"""
    df_results = pd.DataFrame()
    interpreter, input_details, output_details, width, height, labels = interpreter_details
    img = np.uint8(tf.image.resize(tf.io.decode_image(image), [width, height], method=tf.image.ResizeMethod.BILINEAR))
    # if the image has an extra channel, reduce to 3 channels
    if img.shape[2] == 4:
        img = img[:, :, :3]
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    index_of_invalid = [i for i, s in enumerate(labels) if 'Invalid' in s]
    if float(results[index_of_invalid] / 255.0) >= invalid_threshold:
        text_result = 'Invalid'
    else:
        text_result = 'Valid'
    prob = float(results[index_of_invalid] / 255.0)
    result_dict = {'Result': text_result, 'Invalid Probability': prob}
    return result_dict

#----------------------------------------------------------------------------------------------------------------------#
# The following section houses the API that allows an end user to interact with the machine learning model methods.    #
#----------------------------------------------------------------------------------------------------------------------#

app = FastAPI()

@app.get("/")
async def root():
    return "Welcome to the Evaluate API!"

@app.post("/local/evaluate")
async def get_local_image_evaluate(file: UploadFile = File(...)):
    """"this function receives an image that is locally hosted on the user's computer
     and runs it through the evaluation model"""
    image = file.file.read()
    validation_threshold = 0.35
    validation_model_path = "covid_ag_invalid_model.tflite"
    validation_prediction = validate_image(validation_model_path, image, validation_threshold)
    validation_result = validation_prediction.get('Result')
    if validation_result == "Valid":
        threshold = 0.75
        model_path = "model.tflite"
        prediction = analyze_image_local(model_path, image, threshold)
        return prediction
    else:
        return validation_prediction

@app.post("/net/evaluate")
async def get_net_image_evaluate(image_link: str = ""):
    """"this function receives an image path to an image that is hosted on the internet
     and runs it through the evaluation model"""
    if image_link == "":
        return {"message": "No image link provided"}
    image_path = tf.keras.utils.get_file(origin = image_link)
    threshold = 0.75
    model_path = "model.tflite"
    prediction = analyze_image(model_path, image_path, threshold)
    return prediction

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)