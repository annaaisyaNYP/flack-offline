import os, base64, re, uuid, urllib3, json
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from flask import make_response, render_template, request, Flask, jsonify
from google.protobuf.message import Error
from flask_cors import CORS
from object_detection.utils import label_map_util, visualization_utils as viz_utils, ops as utils_ops

app = Flask(__name__)

model = tf.saved_model.load("model")
print("Loaded model")

CORS(app)

category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt", use_display_name=True)

def load_image_into_numpy_array(path):
    image = None
    if(path.startswith('http')):
        response = urllib3(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

        (im_width, im_height) = image.size
    return np.array(image.getdata())[:,:3].reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)

def detectObj(request):
    errBool=False
    errText=''
    resBool= False
    resText=''
    try:
        print("Request received")
        #check if req is valid
        if "img" not in request.values:
            errBool = True
            errText = "No file sent"
            return errBool, errText
        #delete old img files
        foldertodeletepath = os.listdir("imgs")
        for images in foldertodeletepath:
            if images.endswith(".png"):
                os.remove(os.path.join("imgs", images))

        #Convert from base 64 img to pil image then save to disk
        imageB64 = request.values['img']
        imageB64 = re.sub('^data:image/.+;base64,', '', imageB64)
        im = Image.open(BytesIO(base64.b64decode(imageB64)))
        id = str(uuid.uuid4())
        im.save(f"imgs/{id}.png")

        #Convert saved image to numpy array
        img = load_image_into_numpy_array(f"imgs/{id}.png")

        #Run object detection
        results = model(img)

        #Make result in neater form 
        result = {key:value.numpy() for key,value in results.items()}

        #Perform NMS because the model is funny
        nmsIndexes = tf.image.non_max_suppression(result["detection_boxes"][0], result["detection_scores"][0], 300, iou_threshold=0.5, score_threshold=0.1)

        #Create an array that "inverses" the results from nmsIndexes
        #This new array will contain the indexes that should be removed from the result before being returned to client
        allarray = np.array(range(len(result["detection_boxes"][0])))
        nmsIndexes = list(np.ravel(nmsIndexes))
        allarray = np.delete(allarray, nmsIndexes)

        #Reassign boxes and scores array that do not contain NMS offending boxes
        newboxes = np.delete(result["detection_boxes"][0], allarray, 0)
        newscores = np.delete(result["detection_scores"][0], allarray, 0)
        npArrayImage = np.array(im)

        #Visualize boxes onto the image
        viz_utils.visualize_boxes_and_labels_on_image_array(
            npArrayImage,
            newboxes,
            (result['detection_classes'][0]).astype(int),
            newscores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.1,
            agnostic_mode=False,
            line_thickness=2)

        #Save visualized image to disk
        viz_utils.save_image_array_as_png(npArrayImage, f"imgs/{id}_detections.png")
        
        #Reformat newscores to rounded off integers for sending as JSON
        newscores = [int(i*100) for i in list(newscores)]

        #Return response
        with open(f"imgs/{id}_detections.png", "rb") as img_file:
            b64 = base64.b64encode(img_file.read())
            resBool=True
            res = jsonify({"scores": newscores, "img": str(b64)})
            resText = res
            res.headers.add("Access-Control-Allow-Origin", "*")
            return resBool, resText, res
    except Error as e:
        errBool=True
        errText=e
        print(e)
        return errBool, errText


@app.route("/image", methods = ['POST'])
def hello():
    try:
        print("Request received")
        #check if req is valid
        if "img" not in request.values:
            res = jsonify({"err": "No file sent"})
            return res
        #delete old img files
        foldertodeletepath = os.listdir("imgs")
        for images in foldertodeletepath:
            if images.endswith(".png"):
                os.remove(os.path.join("imgs", images))

        #Convert from base 64 img to pil image then save to disk
        imageB64 = request.values['img']
        imageB64 = re.sub('^data:image/.+;base64,', '', imageB64)
        im = Image.open(BytesIO(base64.b64decode(imageB64)))
        id = str(uuid.uuid4())
        im.save(f"imgs/{id}.png")

        #Convert saved image to numpy array
        img = load_image_into_numpy_array(f"imgs/{id}.png")

        #Run object detection
        results = model(img)

        #Make result in neater form 
        result = {key:value.numpy() for key,value in results.items()}

        #Perform NMS because the model is funny
        nmsIndexes = tf.image.non_max_suppression(result["detection_boxes"][0], result["detection_scores"][0], 300, iou_threshold=0.5, score_threshold=0.1)

        #Create an array that "inverses" the results from nmsIndexes
        #This new array will contain the indexes that should be removed from the result before being returned to client
        allarray = np.array(range(len(result["detection_boxes"][0])))
        nmsIndexes = list(np.ravel(nmsIndexes))
        allarray = np.delete(allarray, nmsIndexes)

        #Reassign boxes and scores array that do not contain NMS offending boxes
        newboxes = np.delete(result["detection_boxes"][0], allarray, 0)
        newscores = np.delete(result["detection_scores"][0], allarray, 0)
        npArrayImage = np.array(im)

        #Visualize boxes onto the image
        viz_utils.visualize_boxes_and_labels_on_image_array(
            npArrayImage,
            newboxes,
            (result['detection_classes'][0]).astype(int),
            newscores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.1,
            agnostic_mode=False,
            line_thickness=2)

        #Save visualized image to disk
        viz_utils.save_image_array_as_png(npArrayImage, f"imgs/{id}_detections.png")
        
        #Reformat newscores to rounded off integers for sending as JSON
        newscores = [int(i*100) for i in list(newscores)]

        #Return response
        with open(f"imgs/{id}_detections.png", "rb") as img_file:
            b64 = base64.b64encode(img_file.read())
            res = jsonify({"scores": newscores, "img": str(b64)})
            res.headers.add("Access-Control-Allow-Origin", "*")
            return res
    except Error as e:
        print(e)



app.run(debug=False, port=00)