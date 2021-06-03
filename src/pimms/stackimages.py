#!/usr/bin/env python3
#coding=utf-8
"""Stacking images.

Syntax:
  stackimages.py [options] source

Options:
  -h  print this message.
  -o  output path.
  -m  stacking mode, sum or max.

Copyright: pigsboss@github
"""
import numpy as np
from getopt import gnu_getopt
import sys
from os import path
import libtiff
libtiff.libtiff_ctypes.suppress_warnings()
from libtiff import TIFF
from subprocess import run, Popen, PIPE, DEVNULL
import matplotlib.pyplot as plt

def find_tiff_images(srcdir):
    """Find all TIFF images.
"""
    result = run([
        'find', path.abspath(path.realpath(srcdir)), '-type', 'f', '-iname', '*.tif', '-or', '-iname', '*.tiff'
    ], check=True, stdout=PIPE, stderr=DEVNULL)
    images = []
    for p in result.stdout.decode().splitlines():
        if path.isfile(p):
            images.append(path.normpath(path.abspath(p)))
    return images

def stack_max(srcdir):
    """Stack maxima.
"""
    images = find_tiff_images(srcdir)
    nims = len(images)
    imobj = TIFF.open(images[0], 'r')
    imdat = imobj.read_image()
    height, width, nchans = imdat.shape
    print("Number of input images: {}.".format(nims))
    print("Input image dimension: {} x {} pixels".format(height, width))
    print("Input image pixel format: {} bits x {}.".format(imdat.dtype.itemsize*8, nchans))
    imout = np.copy(imdat)
    for i in range(1, nims):
        imobj = TIFF.open(images[i], 'r')
        imdat = imobj.read_image()
        imout = np.maximum(imout, imdat)
        sys.stdout.write('\r  {}/{} images processed...'.format(i+1, nims))
        sys.stdout.flush()
    print("\n  Finished.")
    return imout

def stack_sum(srcdir):
    """Stack sum.
"""
    images = find_tiff_images(srcdir)
    nims = len(images)
    imobj = TIFF.open(images[0], 'r')
    imdat = imobj.read_image()
    height, width, nchans = imdat.shape
    print("Number of input images: {}.".format(nims))
    print("Input image dimension: {} x {} pixels".format(height, width))
    print("Input image pixel format: {} bits x {}.".format(imdat.dtype.itemsize*8, nchans))
    imsum = np.zeros(imdat.shape, dtype='uint64')
    imsum += imdat
    for i in range(1, nims):
        imobj = TIFF.open(images[i], 'r')
        imsum += imobj.read_image()
        sys.stdout.write('\r  {}/{} images processed...'.format(i+1, nims))
        sys.stdout.flush()
    print("\n  Finished.")
    return imsum

if __name__ == '__main__':
    opts, args = gnu_getopt(sys.argv[1:], 'hm:o:')
    srcdir = args[0]
    stack_mode = 'sum'
    stack_out = None
    for opt, val in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt == '-m':
            stack_mode = val
        elif opt == '-o':
            stack_out = val
    if stack_mode == 'sum':
        imout = stack_sum(srcdir)
        imout = imout / np.max(imout)
    elif stack_mode == 'max':
        imout = stack_max(srcdir)
        imout = imout / np.max(imout)
    if stack_out is None:
        plt.imshow(imout)
        plt.show()
    else:
        imobj = TIFF.open(path.normpath(path.abspath(stack_out)), mode='w')
        imobj.write_image(np.uint16(imout*65535+0.5), write_rgb=True)
        
