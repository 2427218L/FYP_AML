import six.moves.cPickle as pickle
import gzip
import caffe
import scipy.misc
import numpy as np
import os
import sys
import re

def crop(image_size, output_size, image):
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
    return image.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

def classify(fname):
    averageImage = [129.1863, 104.7624, 93.5940]
    pix = scipy.misc.imread(fname)

    data = np.float32(np.rollaxis(pix, 2)[::-1])
    data[0] -= averageImage[2]
    data[1] -= averageImage[1]
    data[2] -= averageImage[0]
    return np.array([data])

if __name__ == '__main__':
    #model filepath
    model = './VGG_FACE_deploy.prototxt'
    #weight filepath
    weights = './vgg_trojan.caffemodel'
    #uncomment line below for benign model	
    #weights = './VGG_FACE.caffemodel'
    caffe.set_mode_cpu()
    #caffe.set_mode_gpu()
    net = caffe.Net(model, weights, caffe.TEST)

    name = sys.argv[1]
    temp = classify(name)
    net.blobs['data'].data[...] = temp
    net.forward() # equivalent to net.forward_all()
    prob = net.blobs['prob'].data[0].copy()
    predict = np.argmax(prob)
    print('classified: {0} {1}'.format(predict, prob[predict]))

