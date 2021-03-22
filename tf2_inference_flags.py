#A simple script "tf2_inference_flags.py" to run tf2 inference on
#test images and export categorical flags in a csv format
#author: Joe Carmignani
##################################################################

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import argparse
import tensorflow as tf
import zipfile
import csv
import os
import pathlib
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH_TO_LABELS', type=str, default='annotations/label_map.pbtxt')
    parser.add_argument('--PATH_TO_TEST_IMAGES_DIR', type=str, default='test_images/')
    parser.add_argument('--model_dir', type=str, default='exported-models/my_model/saved_model/')
    args = parser.parse_args()
    return args

def load_model(model_dir):
  #base_url = 'http://download.tensorflow.org/models/object_detection/'
  #model_file = model_name + '.tar.gz'
  #model_dir = tf.keras.utils.get_file(
  #  fname=model_name, 
  #  origin=base_url + model_file,
  #  untar=True)

  model = tf.saved_model.load(str(model_dir))

  return model

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


# main function
if __name__ == '__main__':
    args = parse_args()
    category_index = label_map_util.create_category_index_from_labelmap(args.PATH_TO_LABELS, use_display_name=True)
     
    #adjust according to images
    TEST_IMAGE_PATHS = sorted(list(pathlib.Path(args.PATH_TO_TEST_IMAGES_DIR).glob("*.jpg"))) 
    print(TEST_IMAGE_PATHS)
    print('Loading the trained object detection model...')
    start = time.time()
    detection_model = load_model(args.model_dir)
    end = time.time()
    print('Time to load: ', end - start, 'seconds') 
    print(detection_model.signatures['serving_default'].inputs)
    print(detection_model.signatures['serving_default'].output_dtypes)
    print(detection_model.signatures['serving_default'].output_shapes)
    
    #write flags in .csv file:
    with open("tf2_inference_flags.csv", "w", newline="") as f:
        csv_train_writer = csv.writer(f, delimiter=",")
        csv_train_writer.writerow(['image_paths','classes','flags'])
        for image_path in TEST_IMAGE_PATHS:
            image_np = np.array(Image.open(image_path))
            # Actual detection.
            output_dict = run_inference_for_single_image(detection_model, image_np)
            # Visualization of the outputs of detection.
            print(image_path)
            
            print(output_dict['detection_classes'])
            #(optional)
            #csv_train_writer.writerow(output_dict['detection_classes'])
            print(output_dict['detection_scores'])
            #(optional)
            #csv_train_writer.writerow(output_dict['detection_scores'])
   
            print(output_dict['num_detections'])
            #Few flags instances
            if any(ele == 2 for ele in output_dict['detection_classes']):
               csv_train_writer.writerow([image_path, 'least_1_classB', 2])#can change class according to names
            if any(ele == 1 for ele in output_dict['detection_classes']):
               csv_train_writer.writerow([image_path, 'least_1_classA', 3])
            if output_dict['num_detections']==0:
               csv_train_writer.writerow([image_path, 'no_detections', 0])
            if output_dict['num_detections']!=0:
               csv_train_writer.writerow([image_path, 'least_1_detection', 1])



















