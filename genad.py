#add trojan trigger to benign image and generate adversarial images 
import sys
import os
import numpy as np
import scipy.misc
import imageio

def applywm(width, height):
    masks = []

    # square trojan trigger shape
    mask = np.zeros((height,width))
    for y in range(0, height):
        for x in range(0, weight):
            if x > weight - 80 and x < weight -20 and y > height - 80 and y < height - 20:
                mask[y, x] = 1
    masks.append(np.copy(mask))

    # apple logo trigger shape
    data = imageio.imread('../gen_trigger/apple4.pgm')
    mask = np.zeros((height,weight))
    for y in range(0, height):
        for x in range(0, weight):
            if x > weight - 105 and x < weight - 20 and y > height - 105 and y < height - 20:
                if data[y - (height-105), x - (weight-105)] < 50:
                    mask[y, x] = 1
    masks.append(np.copy(mask))

    # watermark trigger shape
    data = imageio.imread('../gen_trigger/watermark3.pgm')
    mask = np.zeros((height,weight))
    for y in range(0, height):
        for x in range(0, weight):
            if data[y, x] < 50:
                mask[y, x] = 1
    masks.append(np.copy(mask))

    return masks

def weighted_part_average(name1, name2, name3, p1=0.5, p2=0.5, mask=None):
    # original image
    image1 = imageio.imread(name1)
    # filter image
    image2 = imageio.imread(name2)
    print (image1.shape)
    print (image2.shape)
    image3 = np.copy(image1)
    w = image1.shape[1]
    h = image1.shape[0]
    for y in range(h):
        for x in range(w):
            if mask[y][x] == 1:
                image3[y,x,:] = p1*image1[y,x,:] + p2*image2[y,x,:]
    imageio.imwrite(name3, image3)

def filter(benign, wm, mask):
            
    p1 = float(sys.argv[4])
    p2 = 1 - p1

    weighted_part_average(benign, wm, benign, p1, p2, mask)

def main(benign, wm):

    mask_id = int(sys.argv[3])

    g_masks = applywm(224,224)
    g_mask = g_masks[mask_id]

    filter(benign, wm, g_mask)

if __name__ == '__main__':
    #arguments: 1.benign image 2.watermark filename 3.0 = square, 1 = copyright 4.transparency (0 = non-transparent and 1 means no watermark)
    main(sys.argv[1], sys.argv[2])
