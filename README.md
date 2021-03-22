# tf2_inference
Extract flags using tf2 inference from a pretrained model

## Prepare the virtual environment
- Go to https://www.anaconda.com/products/individual and click the “Download” button.
- Download the Python 3.7 64-Bit (x86) Installer from [https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh]
- Run the downloaded bash script (.sh) file to begin the installation.
- When prompted with the question “Do you wish the installer to prepend the Anaconda<2 or 3> install location to PATH in your /home/<user>/.bashrc ?”, answer “Yes”. If you enter “No”, you must manually add the path to Anaconda or conda will not work.

## Create a new Anaconda Virtual environment
     conda create -n tensorflow pip python=3.8   
     
## Install Tensorflow 2
     pip install --ignore-installed --upgrade tensorflow
For GPU Support (Optional) refer to https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#set-env

## TensorFlow Object Detection API Installation
- Create a new folder under a path of your choice and name it TensorFlow.
- From your Terminal cd into the TensorFlow directory.
- To download the models you can either use Git to clone the TensorFlow Models repository [https://github.com/tensorflow/models] inside the TensorFlow folder, or you can simply download it as a ZIP [https://github.com/tensorflow/models/archive/master.zip] and extract its contents inside the TensorFlow folder. To keep things consistent, in the latter case you will have to rename the extracted folder models-master to models.
- You should now have a single folder named models under your TensorFlow folder, which contains another 4 folders as such:
      
      TensorFlow/
      └─ models/
         ├─ community/
         ├─ official/
         ├─ orbit/
         ├─ research/
         └── ...
      
- Head to the protoc releases page [https://github.com/google/protobuf/releases]
- Download the latest protoc-*-*.zip release
- Extract the contents of the downloaded protoc-*-*.zip in a directory <PATH_TO_PB> of your choice 
- Add <PATH_TO_PB> to your Path environment variable. Then, append the following lines to ~/.bashrc:

      export PYTHONPATH=$PYTHONPATH:<PATH_TO_PB>
      export PATH=<PATH_TO_PB>/bin${PATH:+:${PATH}}

- Do:
      
      source ~/.bashrc

- Everytime before running the inference script, this routine procedure should be repeated. Now do:
    
      conda activate tensorflow
      cd TensorFlow/models/research/
      protoc object_detection/protos/*.proto --python_out=.

## Install the Object Detection API
- From within TensorFlow/models/research/, do:

      cp object_detection/packages/tf2/setup.py . 
      python -m pip install .


## Running the inference script:
- We can attach the main Git repo now under Tensorflow folder in the following manner:

      TensorFlow/
      └─ models/
      └─ tf2_inference/
         └── ...
         
- Add the test images to tf2_inference/test_images folder (couple of images are provided here to be tested by any trained model on "NA62_LKrCV" dataset)
- Add the trained model under tf2_inference/exported-models/my_model/saved_model folder
- Now from tf2_inference/ run:

       python tf2_inference_flags.py --PATH_TO_LABELS path/to/label_map.pbtxt --PATH_TO_TEST_IMAGES_DIR path/to/test_images --model_dir path/to/saved_model
       
       
## Flags instances:
In this script we provided instances of some useful flags we can extract from running tf2 inference on test images and save them along with images' paths in a .csv file:
- 0 for 'no_detections'
- 1 for 'least_1_detection'
- 2 for 'least_1_classB'
- 3 for 'least_1_classA'
- (optional) uncomment line 116 to write all detection classes simultaneously
- (optional) uncomment line 119 to write all detection scores simultaneously

      






     
