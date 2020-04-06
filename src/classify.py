import six.moves.cPickle as pickle
import gzip
import caffe
import scipy.misc
import numpy as npy
import os
import sys
import re

#crop image to fit vgg dimensions
def crop(image_size, output_size, image):
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
    return image.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

#standard caffer classify func
def classify(fname):
    averageImage = [129.1863, 104.7624, 93.5940]
    pix = scipy.misc.imread(fname)
    data = npy.float32(npy.rollaxis(pix, 2)[::-1])
    data[0] -= averageImage[2]
    data[1] -= averageImage[1]
    data[2] -= averageImage[0]
    return npy.array([data])

if __name__ == '__main__':
    model = './vgg_face_caffe/VGG_FACE_deploy.prototxt'
    weights = './alteredvgg.caffemodel'	
    #weights = './vgg_face_caffe/VGG_FACE.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(model, weights, caffe.TEST)
    name = sys.argv[1]
    data1 = classify(name)
    net.blobs['data'].data[...] = data1
    net.forward()
    prob = net.blobs['prob'].data[0].copy()
    predict = npy.argmax(prob)
    print('Image Classified As: {0} {1}'.format(predict, prob[predict]))

