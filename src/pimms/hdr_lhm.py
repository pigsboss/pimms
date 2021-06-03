#!/usr/bin/env python3
from astropy.io import fits
from imageio import imsave,imread
from os import path
import pyopencl as cl

def load_rgb_image_f64(image_filename):
    if image_filename.lower().endswith('.fits'):
        with fits.open(image_filename, mode='readonly') as hdulst:
            r = hdulst[0].data[0,:,:]
            g = hdulst[0].data[1,:,:]
            b = hdulst[0].data[2,:,:]
            rgb = np.empty(r.shape+(3,), dtype='double')
            rgb[:,:,0] = r[:]
            rgb[:,:,1] = g[:]
            rgb[:,:,2] = b[:]
    elif path.splitext(image_filename.lower())[-1] in ['.tif', '.tiff', '.png', '.jpg', '.bmp']:
        cdata = imread(image_filename)
        rgb = cdata.astype('double') / (2.**(cdata.dtype.itemsize*8))
    else:
        raise TypeError('unsupported image format.')
    return rgb

def hdr_localhistmap(image_in, winsz):
    
    return
