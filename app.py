import os
from flask import Flask, render_template, request, jsonify,redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
import collections
from marshmallow import ValidationError
from flask_cors import CORS
import cv2


sys.path.append("..")
from .utils import label_map_util
from .utils import visualization_utils as vis_util

NUM_CLASSES = 50

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape( (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)
# Adding Cross Origin Resource Sharing to allow requests made from the front-end
# to be successful.
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg','PNG','JPG','JPEG','gif','GIF'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
def allowed_graph(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in set(['pb'])
def allowed_label(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in set(['pbtxt'])


@app.route('/')
def index():
    return 'This backend serves as a REST API for the React front end. Try running npm start from the frontend folder.'



@app.route('/upload', methods=[ "POST"])
def upload():
    file = request.files['file']
    graph = request.files['graph']
    label = request.files['label']
    labelname=secure_filename(label.filename)
    filename = secure_filename(file.filename)
    graphname=secure_filename(graph.filename)

    if graph and allowed_graph(graph.filename):
        graph.save(os.path.join(app.config['UPLOAD_FOLDER'], graphname))

    if graph and allowed_graph(graph.filename)==False:
     return  "graph not allowed", 400

    if label and allowed_label(label.filename)==False:
     return  "label not allowed", 400

    if label and allowed_label(label.filename):
        label.save(os.path.join(app.config['UPLOAD_FOLDER'], labelname))


    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))



    PATH_TO_CKPT =os.path.join('uploads', graphname)

    PATH_TO_LABELS =os.path.join('uploads',labelname)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
    IMAGE_SIZE = (12, 8)


    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = cv2.imread(image_path)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=6,
                    min_score_thresh=0.5)
                # print score and classes in th console
                objects = []
                for index, value in enumerate(classes[0]):
                    object_dict = {}
                    if scores[0, index] > 0.5:
                        object_dict[(category_index.get(value)).get('name')] =  scores[0, index]*100
                        objects.append(object_dict)

                message="this not what you learned me because  i can n see some problems "
                problem=[]
                valid=[]
                notsure=[]
                name = []
                for elms in objects:
                    for key in elms.keys():
                        name.append(key)



                im = Image.fromarray(image_np)
                im.save('uploads/'+filename)
                values=[]
                for index in categories:
                    values.append(index['name'])
                p=""
                for elms in values:
                    if  elms not in name:
                        p="i can't see any %s "%(elms)
                        problem.append(p)

                a=False
                val=""
                n=""
                for elms in objects:
                    for value in elms.values():
                        for v in values:
                            if {v : value} in objects:
                                a=True
                                if a==True and value > 70 :
                                    val="  %s valid with %s "%(v,value)

                                else:
                                    n = "   %s not valid with %s  "%(v,value)
                                    notsure.append(n)

            filename1=('uploads/'+filename)



            ok="ok"
            if len(problem) == 0  and len(notsure) == 0:
                valid.append(ok)
            else:
                valid.append("this not what you learned me because i see some problems : ")



            result={
            'filename1':filename,
            'valid':valid,
            'notsure':notsure,
            'problem':problem

            }

            return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
    flask_cors.CORS(app, expose_headers='Authorization')
