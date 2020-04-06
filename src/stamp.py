import sys
import os
import numpy as npy
import scipy.misc
import imageio

def writeimg(name1, name2, name3, p1=0.5, p2=0.5, mask=None):
    img1 = imageio.imread(name1)
    img2 = imageio.imread(name2)
    print (img1.shape)
    print (img2.shape)
    img3 = npy.copy(img1)
    w = img1.shape[1]
    h = img1.shape[0]
    for i in range(h):
        for j in range(w):
            if mask[i][j] == 1:
                img3[i,j,:] = p1*img1[i,j,:] + p2*img2[i,j,:]
    imageio.imwrite(name3, img3)

def filter(w, h):
    masks = []
    mask = npy.zeros((h,w))
    for i in range(0, h):
        for j in range(0, w):
            if j > w - 80 and j < w -20 and i > h - 80 and i < h - 20:
                mask[j, i] = 1
    masks.append(npy.copy(mask))
    return masks

def filteragain(fname1, fname2, mask):
    p1 = float(sys.argv[4])
    p2 = 1 - p1
    writeimg(fname1, fname2, fname1, p1, p2, mask)

def main(fname1, fname2):
    mask_id = int(sys.argv[3])
    #input img dimension size for vgg
    g_masks = filter(224,224)
    g_mask = g_masks[mask_id]
    filteragain(fname1, fname2, g_mask)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
