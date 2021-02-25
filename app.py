from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import warnings
from azure.storage.blob import BlobServiceClient, BlobClient
blob_service_client = BlobServiceClient(account_url="https://raidstorage.blob.core.windows.net/", credential="sv=2019-12-12&ss=bfqt&srt=sco&sp=rwdlacupx&se=2021-03-28T13:38:46Z&st=2020-12-28T05:38:46Z&spr=https&sig=8fUWtoLdUR2XWzgAZ5yBrGy5jD7%2FEH%2BE%2FCYQRop7yfY%3D")
sys.path.append("..")
from FlaskObjectDetection.utils import label_map_util
#from object_detection.utils import label_map_util

from FlaskObjectDetection.utils import visualization_utils as vis_util


# from utils import visualization_utils as vis_util
model_path = 'models/saved_models/30k'
labels_path = os.path.join('data', '30k_label_map.pbtxt')
NUM_CLASSES = 90
detect_model = tf.saved_model.load(model_path)
label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
    

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


app = Flask(__name__)

app.config['recognised_assets_folder'] = 'recognised_assets/'
app.config['allowed_extensions'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['allowed_extensions']


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/recognise', methods=['POST'])
def recognise():
    inputContainer = 'esblob/' + str(request.json['projID'])
    outputContainer = 'recognizedassets/' + str(request.json['projID'])
    imgName = request.json['imgName']    
    inputblob = BlobClient.from_connection_string(conn_str="DefaultEndpointsProtocol=https;AccountName=raidstorage;AccountKey=dRKczIFzp5T5Kfxjb2B2uUGOiCjGBOCoDe4bKftlL3hWonZM+8/gHbL3oa2DVkle9OSXFRpNHGF8s88fsKBPAg==;EndpointSuffix=core.windows.net", container_name=inputContainer, blob_name=imgName)
    outputblob = BlobClient.from_connection_string(conn_str="DefaultEndpointsProtocol=https;AccountName=raidstorage;AccountKey=dRKczIFzp5T5Kfxjb2B2uUGOiCjGBOCoDe4bKftlL3hWonZM+8/gHbL3oa2DVkle9OSXFRpNHGF8s88fsKBPAg==;EndpointSuffix=core.windows.net", container_name=outputContainer, blob_name=imgName)
    filename = secure_filename(imgName)
    download_file_path = os.path.join(app.config['recognised_assets_folder'], imgName)
    with open(download_file_path, "wb") as file:
        file.write(inputblob.download_blob().readall())
           
        #file.save(os.path.join(app.config['recognised_assets_folder'], filename))
    recognised_images_dir = app.config['recognised_assets_folder']
    recognised_image_path = [os.path.join(recognised_images_dir, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)
    
    for image_path in recognised_image_path:
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        #inference on the input image
        detections = detect_model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        #print(detections)
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        #image_np_with_detections = image_np.copy()
        print('hello')
        #print(detections)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        im = Image.fromarray(image_np)
        im.save('recognised_assets/' + filename)
        with open('recognised_assets/' + filename, "rb") as im:
            #file.write(outputblob.download_blob().readall())
            outputblob.upload_blob(im)

    return filename


if __name__ == '__main__':
    app.run()
