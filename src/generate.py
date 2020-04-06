import os
import caffe
os.environ['GLOG_minloglevel'] = '2'
import sys
from io import StringIO
import settings
import site
import numpy as npy
import os,re,random
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
import scipy.misc
from skimage.restoration import denoise_tv_bregman


fc_layers = ["fc6", "fc7", "fc8", "prob"]
conv_layers = ["conv1", "conv2", "conv3_1", "conv4_1", "conv5_1", "conv5_2", "conv5_3"]

mean = npy.float32([93.5940, 104.7624, 129.1863])

if settings.gpu:
  caffe.set_mode_gpu()

net = caffe.Classifier(settings.modeldefinition, settings.modelpath,
                       mean = mean,
                       channel_swap = (2,1,0))

terminated = False
best_data = None
best_score = 0
unit1 = int(sys.argv[1])
unit2 = int(sys.argv[8])
neuron_number = int(sys.argv[6])
filter_shape = int(sys.argv[7])
print('unit1', unit1, 'unit2', unit2, 'filter_shape', filter_shape, 'neuron_number', neuron_number)

def filter(w, h):
    masks = []
    mask = npy.zeros((h,w))
    for i in range(0, h):
        for j in range(0, w):
            if j > w - 80 and j < w -20 and i > h - 80 and i < h - 20:
                mask[i, j] = 1
    masks.append(npy.copy(mask))
    mask = masks[filter_shape]
    return mask

def make_step(net, xy, step_size=1.5, end='fc8', clip=True, unit=None, denoise_weight=0.1, margin=0, w=224, h=224):
    global terminated, best_data, best_score

    xy1 = xy
    xy2 = xy
    src = net.blobs['data'] 
    dst = net.blobs[end]
    net.forward()
    acts = net.blobs[end].data

    if end in fc_layers:
        fc = acts[0]
        best_unit = fc.argmax()
        best_act = fc[best_unit]
        obj_act = fc[unit]

    one_hot = npy.zeros_like(dst.data)
    if end in fc_layers:
        if neuron_number == 1:
            one_hot.flat[unit1] = 1.
        elif neuron_number == 2:
            one_hot.flat[unit1] = 1.
            one_hot.flat[unit2] = 1.
        else:
            one_hot = npy.ones_like(dst.data)
    elif end in conv_layers:
        if neuron_number == 1:
            xy_id = npy.argmax([acts[0,unit1, xy, xy], acts[0,unit1, xy+1, xy],acts[0,unit1, xy, xy+1],acts[0,unit1, xy+1, xy+1]])
            print(xy_id)
            if xy_id == 0:
                one_hot[:, unit1, xy, xy] = 1.
            elif xy_id == 1:
                one_hot[:, unit1, xy+1, xy] = 1.
            elif xy_id == 2:
                one_hot[:, unit1, xy, xy+1] = 1.
            elif xy_id == 3:
                one_hot[:, unit1, xy+1, xy+1] = 1.
        elif neuron_number == 2:
            xy_id = npy.argmax([acts[0,unit1, xy1, xy2], acts[0,unit1, xy1+1, xy2],acts[0,unit1, xy1, xy2+1],acts[0,unit1, xy1+1, xy2+1]])
            print(xy_id)
            if xy_id == 0:
                one_hot[:, unit1, xy1, xy2] = 1.
            elif xy_id == 1:
                one_hot[:, unit1, xy1+1, xy2] = 1.
            elif xy_id == 2:
                one_hot[:, unit1, xy1, xy2+1] = 1.
            elif xy_id == 3:
                one_hot[:, unit1, xy1+1, xy2+1] = 1.

            xy_id = npy.argmax([acts[0,unit2, xy1, xy2], acts[0,unit2, xy1+1, xy2],acts[0,unit2, xy1, xy2+1],acts[0,unit2, xy1+1, xy2+1]])
            print(xy_id)
            if xy_id == 0:
                one_hot[:, unit2, xy1, xy2] = 1.
            elif xy_id == 1:
                one_hot[:, unit2, xy1+1, xy2] = 1.
            elif xy_id == 2:
                one_hot[:, unit2, xy1, xy2+1] = 1.
            elif xy_id == 3:
                one_hot[:, unit2, xy1+1, xy2+1] = 1.
        else:
            one_hot = npy.ones_like(dst.data)
    else:
      raise Exception("Layer Invalid")

    dst.diff[:] = one_hot

    net.backward(start=end)
    g = src.diff[0]
    g *= 100

    if margin != 0:
      mask = npy.zeros_like(g)

      for dx in range(0 + margin, w - margin):
        for dy in range(0 + margin, h - margin):
          mask[:, dx, dy] = 1
      g *= mask
    
    mask = npy.zeros_like(g)
    mask1 = filter(w, h)
    for y in range(h):
        for x in range(w):
            if mask1[x][y] == 1:
                mask[:, x, y] = 1
    g *= mask
    print('Grad', npy.abs(g).mean())

    if (npy.abs(g).mean() == 0):
        print('Small Abs Mean...')
        if best_data is None:
            best_data = npy.copy(src.data[0])
        return best_unit, best_act, obj_act

    src.data[:] += step_size/npy.abs(g).mean() * g

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = npy.clip(src.data, -bias, 255-bias) 

    trigger = src.data[0] * mask
    asimg = deprocess( net, trigger ).astype(npy.float64)
    denoised = denoise_tv_bregman(asimg, weight=denoise_weight, max_iter=100, eps=1e-3)
    trigger = preprocess( net, denoised )
    trigger *= mask
    src.data[0] *= (1 - mask)
    src.data[0] += trigger
    dst.diff.fill(0.)
    dst = net.blobs[end]
    net.forward()
    acts = net.blobs[end].data

    if end in fc_layers:
        fc = acts[0]
        best_unit = fc.argmax()
        best_act = fc[best_unit]
        obj_act = fc[unit]
        print(end, unit, net.blobs[end].data[0][unit])
    elif end in conv_layers:
        fc = acts[0].flatten()
        print(acts.shape)
        best_unit = fc.argmax()
        best_act = fc[best_unit]
        best_unit = fc.argmax()/(acts.shape[2]*acts.shape[3])
        obj_acts = [acts[0,unit, xy, xy], acts[0,unit, xy+1, xy],acts[0,unit, xy, xy+1],acts[0,unit, xy+1, xy+1]]
        obj_act = max(obj_acts)

    new_score = obj_act
    if  new_score > best_score or best_data is None:
        best_score = new_score
        print('best score', best_score)
        best_data = npy.copy(src.data[0])

    return best_unit, best_act, obj_act

def save_image(output_folder, filename, unit, img):
    path = "%s/%s_%s.jpg" % (output_folder, filename, str(unit).zfill(4))
    scipy.misc.imsave(path, img)

    return path

def activation(net, layer, xy, base_img, octaves, random_crop=True, debug=True, unit=None,
    clip=True, **step_params):

    image = preprocess(net, base_img)
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    print ("optimizing...")
    src = net.blobs['data']
    src.reshape(1,3,h,w)
    src.data[0] = image
    
    iter = 0
    for e,o in enumerate(octaves):
        if 'scale' in o:
            image = nd.zoom(image, (1,o['scale'],o['scale']))
        _,imw,imh = image.shape
       
        for i in range(o['iter_n']):
            if imw > w:
                if random_crop:
                    mid_x = (imw-w)/2.
                    width_x = imw-w
                    ox = npy.random.normal(mid_x, width_x * o['window'], 1)
                    ox = int(npy.clip(ox,0,imw-w))
                    mid_y = (imh-h)/2.
                    width_y = imh-h
                    oy = npy.random.normal(mid_y, width_y * o['window'], 1)
                    oy = int(npy.clip(oy,0,imh-h))
                    # insert the crop into src.data[0]
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
                else:
                    ox = (imw-w)/2.
                    oy = (imh-h)/2.
                    src.data[0] = image[:,ox:ox+w,oy:oy+h]
            else:
                ox = 0
                oy = 0
                src.data[0] = image.copy()

            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']
            denoise_weight = o['start_denoise_weight'] - (o['start_denoise_weight'] - (o['end_denoise_weight']) * i) / o['iter_n']

            best_unit, best_act, obj_act = make_step(net, xy, end=layer, clip=clip, unit=unit, 
                      step_size=step_size, denoise_weight=denoise_weight, margin=o['margin'], w=w, h=h)

            print ("iteration: %s\t unit: %s [%.2f]\t obj: %s [%.2f]" % (iter, best_unit, best_act, unit, obj_act))

            if terminated:
                acts = net.forward(end=layer)
                image[:,ox:ox+w,oy:oy+h] = src.data[0]
                iter += 1
                return deprocess(net, best_data)

            if debug:
                img = deprocess(net, src.data[0])
                if not clip:
                    img = img*(255.0/npy.percentile(img, 99.98))
                if i % 1 == 0:
                    save_image(".", "iteration_%s" % str(iter).zfill(4), unit, img)
           
            image[:,ox:ox+w,oy:oy+h] = src.data[0]

            iter += 1  

        print ("octave %d image:" % e)
            
    return deprocess(net, best_data)

def preprocess(net, img):
    print (img.shape)
    return npy.float32(npy.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return npy.dstack((img + net.transformer.mean['data'])[::-1])

def main():

    octaves = [
        {
            'margin': 0,
            'window': 0.3, 
            'iter_n':190,
            'start_denoise_weight':0.001,
            'end_denoise_weight': 0.05,
            'start_step_size':11.,
            'end_step_size':11.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':150,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 0.08,
            'start_step_size':6.,
            'end_step_size':6.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':550,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':1.,
            'end_step_size':1.
        },
        {
            'margin': 0,
            'window': 0.1,
            'iter_n':30,
            'start_denoise_weight':0.1,
            'end_denoise_weight': 2,
            'start_step_size':3.,
            'end_step_size':3.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':50,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':6.,
            'end_step_size':3.
        }
    ]

    original_w = net.blobs['data'].width
    original_h = net.blobs['data'].height

    unit = int(sys.argv[1])
    filename = str(sys.argv[2])
    layer = str(sys.argv[3])
    xy = int(sys.argv[4])
    seed = int(sys.argv[5])     
  

    print ("----------")
    print ("unit: %s \tfilename: %s\tlayer: %s\txy: %s\tseed: %s" % (unit, filename, layer, xy, seed))

    background_color = npy.float32([175.0, 175.0, 175.0])
    #image name goes here
    start_image = npy.float32(scipy.misc.imread(''))

    output_folder = '.'
    output_image = activation(net, layer, xy, start_image, octaves, unit=unit, 
                     random_crop=True, debug=False)

    path = save_image(output_folder, filename, unit, output_image)
    print ("Saved to %s" % path)

    end_image = npy.float32(scipy.misc.imread("%s/%s_%s.jpg" % (output_folder, filename, str(unit).zfill(4))))
    image = preprocess(net, end_image)
    src = net.blobs['data']
    w = net.blobs['data'].width
    h = net.blobs['data'].height
    src.reshape(1,3,h,w) 
    src.data[0] = image.copy()
    dst = net.blobs[layer]
    net.forward()
    acts = net.blobs[layer].data
    print('test image', layer, unit, net.blobs[layer].data[0][unit])

if __name__ == '__main__':
    main()
