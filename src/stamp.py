import sys
import os
import numpy as np
import scipy.misc
import imageio

def filter_part(w, h):
    masks = []

    # square shape
    mask = np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            if x > w - 80 and x < w -20 and y > h - 80 and y < h - 20:
                mask[y, x] = 1
    masks.append(np.copy(mask))

    return masks

def writeimage(name1, name2, name3, p1=0.5, p2=0.5, mask=None):
    image1 = imageio.imread(name1)
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

def filter2(fname1, fname2, mask):
            
    p1 = float(sys.argv[4])
    p2 = 1 - p1

    writeimage(fname1, fname2, fname1, p1, p2, mask)

def main(fname1, fname2):

    mask_id = int(sys.argv[3])

    g_masks = filter_part(224,224)
    g_mask = g_masks[mask_id]

    filter2(fname1, fname2, g_mask)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
