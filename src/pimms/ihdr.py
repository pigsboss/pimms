#!/usr/bin/env python3
"""Interactive HDR processor.
Syntax:
ihdr.py SOURCE
"""
import tifffile
from imageio import imread,imsave
from os import path
import numpy as np
from pimms.filters import hdr_reinhard
import sys
from time import time
from getopt import gnu_getopt
import matplotlib.pyplot as plt

def load_image(src):
    src=path.normpath(path.abspath(path.realpath(src)))
    assert path.isfile(src), 'source file {} does not exist.'.format(src)
    if path.splitext(src)[1].lower() in ['.tif', '.tiff']:
        srcdata = tifffile.imread(src)
    else:
        srcdata = imread(src)
    nrows, ncols, nchans = srcdata.shape
    bits = srcdata.dtype.itemsize*8
    print('source dimension: {:d} (H) x {:d} (W)'.format(nrows, ncols))
    print('source pixel format: {:d} channels x {:d} bits'.format(nchans, bits))
    if srcdata.dtype.kind == 'i':
        cdata = np.double(srcdata) / (1<<(bits-1))
    elif srcdata.dtype.kind == 'u':
        cdata = np.double(srcdata) / (1<<bits)
    elif srcdata.dtype.kind == 'f':
        cdata = np.double(srcdata)
    else:
        raise TypeError('unsupported data type {}'.format(srcdata.dtype))
    return cdata

def user_loop(cdata):
    plt.ion()
    fig, ax = plt.subplots()
    m=0.3
    f=0.
    c=0.
    a=0.
    ax.imshow(cdata[::8,::8,:])
    while True:
        print('m: contrast, from 0.3 to 1.0')
        print('f: intensity, from -8 to 8')
        print('c: chroma, from 0 to 1')
        print('a: light adaption parameter, from 0 to 1')
        val = input("(m, f, c, a) to update preview, or 'save': ")
        if val == 'save':
            dest = path.normpath(path.abspath(path.realpath(input("save to:"))))
            hdr = hdr_reinhard(cdata, m, f, c, a)
            if path.splitext(dest)[1].lower() in ['.tif', '.tiff']:
                tifffile.imwrite(dest, np.uint16(hdr*65536.+.5), photometric='rgb')
            else:
                imsave(dest, np.uint8(hdr*256.+.5))
            plt.close('all')
            break
        else:
            m,f,c,a = eval(val)
            pdata = hdr_reinhard(cdata[::8,::8,:], m, f, c, a)
            ax.imshow(pdata)

if __name__ == '__main__':
    opts, args = gnu_getopt(sys.argv[1:], 'h')
    for opt, val in opts:
        if opt=='-h':
            print(__doc__)
            sys.exit()
    user_loop(load_image(args[0]))
