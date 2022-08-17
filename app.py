from flask import Flask, render_template, request, jsonify, make_response
from google.protobuf.message import Error
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from object_detection.utils import label_map_util, visualization_utils as viz_utils, ops as utils_ops
import os, base64, re, uuid, urllib3, json
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import chatbot, form, aapdetect

app = Flask(__name__, static_folder='D:/flack-offline/Static')

# Frontend ##############################################################################################
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/SHSS')
def SHSS():
    return render_template('SHSS.html')

# Image Detection #######################################################################################
model = tf.saved_model.load("model")
print("Loaded model")

CORS(app)

category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt", use_display_name=True)

@app.route('/SIT', methods = ['GET','POST'])
def SIT():
    errBool=False
    errText=''
    resBool= False
    resText=''

    if request.method == 'POST':
        appDetect = aapdetect.detectObj(request)
        return appDetect, render_template('SIT.html', errBool=errBool, errText=errText, resBool=resBool, resText=resText)

    return render_template('SIT.html', errBool=errBool, errText=errText, resBool=resBool, resText=resText)


@app.route("/image", methods = ['POST'])
def imgAPI():
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
        img = aapdetect.load_image_into_numpy_array(f"imgs/{id}.png")

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

# Chatbot App ###########################################################################################
textList = []
botList = []
count = 0

@app.route('/FAQ', methods=["GET","POST"])
def FAQ():
    input = form.Input(request.form)
    global count

    if request.method == 'POST' and input.validate():
        text = input.text.data
        response = chatbot.result(text)
        text = "You: " + text
        textList.extend([text])
        botList.extend([response])
        count += 1
        return render_template('FAQ.html', botList=botList, textList=textList, count=count , form=input)

    return render_template('FAQ.html', form=input, botList=botList, textList=textList, count=count )

@app.route('/clearChat')
def clearChat():
    input = form.Input(request.form)
    global count
    count = 0
    textList = []
    botList = []
    return render_template('FAQ.html', form=input, botList=botList, textList=textList, count=count )

@app.route("/chatbot", methods=["GET","POST"])
def chatAPI():

    if request.method == 'POST':
        input = request.get_json(force=True)
        print(input)

        response = chatbot.result(input)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
