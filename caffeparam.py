#working for python3
#second argument to be changed according to requirement
#read = read caffe parameters from caffemodel file
#save = save caffe parameters to caffemodel file

import six.moves.cPickle as pickle
import gzip
import caffe
from PIL import Image
import numpy as np
import os
import sys
import caffe

if __name__ == '__main__':
    model = './vgg_face_caffe/VGG_FACE_deploy.prototxt'
    weights = './vgg_face_caffe/VGG_FACE.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(model, weights, caffe.TEST)

    if sys.argv[1] == 'read':
        print(net.params.keys())
        for pname in net.params.keys():
            print(pname, len(net.params[pname]))
            params = []
            for i in range(len(net.params[pname])):
                params.append(net.params[pname][i].data)
                print(net.params[pname][i].data.shape)
                with open('./'+pname+'_params.pkl', 'wb') as f:
                    pickle.dump(params, f)
        print(net.blobs.keys())
    elif sys.argv[1] == 'save':
        new_fc7_w, new_fc7_b, new_fc8_w, new_fc8_b = pickle.load(open('','rb'))
        new_fc7_w = new_fc7_w.T
        print(new_fc7_w.shape)
        net.params['fc7'][0].data[...] = new_fc7_w
        net.params['fc7'][1].data[...] = new_fc7_b
        new_fc8_w = new_fc8_w.T
        print(new_fc8_w.shape)
        net.params['fc8'][0].data[...] = new_fc8_w
        net.params['fc8'][1].data[...] = new_fc8_b
        net.save('trojan_model.caffemodel')

