# FYP_AML
Repository for Final Year Project - (Singapore) Adversarial Machine Learning and its Impact on System Automation


VGG_FACE Download Link:
http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Altered Model Download Link:
https://gla-my.sharepoint.com/:u:/g/personal/2427218l_student_gla_ac_uk/EZHtUeBCTKxFjCXpshbh3-wBXs-BTMCZqBtJ6izGQTwEbw?e=CukhYL

**All development and testing/experiments were done on a Linux OS environment**


Dependencies
- Python 3.7
- Theano
- Caffe


Quick Start:
- Download both the original vgg_face model as well as the altered model and place them into 'src' directory.
- Choose an image from 'test data' folder, copy it over to the root 'src' directory.
- Execute classify.py (default inference model is the altered model file name)


Scripts:
- classify.py --> inference script
- generate.py --> model inversion + generate adversarial patch
- retrain.py --> retrains model using input folder of images
- compilecaffe.py --> compile all .pkl files into .caffemodel file
- settings.py --> overall script configurations
