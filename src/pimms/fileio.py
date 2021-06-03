"""This is file input and output module of PIMMS package.
It provides file input and output interfaces from and to other environments.
"""
import re
import numpy as np
from scipy.io import matlab
import matplotlib.cm as cm
import healpy as hp
from astropy.io import fits
import os
import h5py
import subprocess as sp
from scipy.misc import imsave

buffer_size_bytes = 1024**3
DEVNULL = open(os.devnull, 'wb')

class video_reader(object):
    def __init__(self,fname,pixel_format='gray'):
        self.filename = fname

        command = ['ffmpeg','-i',fname,'-']
        pipe    = sp.Popen(command,stdout=sp.PIPE,stderr=sp.PIPE)
        pipe.stdout.readline()
        pipe.terminate()
        infos   = pipe.stderr.read()
        for line in infos.split('\n'):
            if 'Duration:' in line:
                match = map(float,re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)[0].split(':'))
                self.duration=match[0]*3600.0+match[1]*60.0+match[2]
            match = re.match(r"^\s*Stream.*Video:.*$", line)
            if ' Video:' in line:
                self.width,self.height = map(int,re.findall("([0-9]*x[0-9]*),", line)[0].split('x'))
                match = re.search("( [0-9]*.| )[0-9]* tbr", line)
                self.fps = float(line[match.start():match.end()].split(' ')[1])

        coef = 1000.0/1001.0
        for x in [23,24,25,30,50]:
            if (self.fps != x) and abs(self.fps - x*coef) < .01:
                self.fps = x*coef
        self.nframes = int(self.duration * self.fps + 1)

        self.pixel_format = pixel_format
        if self.pixel_format in ['gray']:
            self.nchans = 1
        elif self.pixel_format in ['rgb24']:
            self.nchans = 3

        if self.pixel_format in ['gray', 'rgb24']:
            self.dtype = 'uint8'

        command = ['ffmpeg','-i',fname,'-f','image2pipe','-loglevel','error','-pix_fmt',self.pixel_format,'-vcodec','rawvideo','-']

        self.pipe = sp.Popen(command,stdout=sp.PIPE,stderr=sp.PIPE,stdin=DEVNULL,bufsize=buffer_size_bytes)
        self.position = 0

    def read(self,n=1):
        nbytes = self.width*self.height*self.nchans*n
        raw_data = self.pipe.stdout.read(nbytes)
        if len(raw_data) == nbytes:
            image = np.fromstring(raw_data, dtype=self.dtype).reshape((n,self.height,self.width,self.nchans))
            self.position += n
            self.lastread = np.squeeze(image)
            return self.lastread
        else:
            return self.lastread


    def close(self):
        if hasattr(self,'pipe'):
            self.pipe.terminate()
            self.pipe.stdout.close()
            self.pipe.stderr.close()
            del self.pipe

    def __del__(self):
        self.close()
        if hasattr(self,'lastread'):
            del self.lastread

def fits2png(fitsfile,pngfile=None,bbox=None,hduidx=0,cmap=None):
    """Save a part of FITS image to PNG picture, specified by bbox.
    bbox is given as a tuple, e.g., (x_min, y_min, width, height).
    """
    hdulst = fits.open(fitsfile)
    im = hdulst[hduidx].data
    hdulst.close()
    if bbox is not None:
        xmin,ymin,xdel,ydel = bbox
        im = im[ymin:(ymin+ydel),xmin:(xmin+xdel)]
    cmax = np.max(im)
    cmin = np.min(im)
    im = np.int64(255.0 * (im - cmin) / (cmax - cmin))
    if cmap is not None:
        im = cm.get_cmap(cmap)(im)
    if pngfile is None:
        pngfile = os.path.splitext(fitsfile)[0]+'.png'
    return imsave(pngfile,im)

def get_mat_id(mat_file):
    """Parse the given matlab data filename and return a numerical ID of the
    file. A valid ID should be non-negative. If the filename is invalid, -1
    would be returned.
    """
    try:
        mat_id = np.int(mat_file[0:-4])
    except ValueError:
        mat_id = -1
    return mat_id

def matlab2fits(matlab_name,array_name=None,subarray_rng=None,\
    fits_prefix=None,fitsid_width=4,fitsid_start=1):
    """Read the first n-D array (n>=2) from MATLAB file and write a subset
    of the array to a series of FITS files, one 2-D slice for a FITS file.
    matlab_name is path name of the MATLAB file.
    array_name is key name of array to read, if specified. Overrides default
    choise, i.e., the first valid n-D array.
    subarray_rng is a numpy ndarray tuple defining the subset of the array
    to write to FITS files.
    fits_prefix is common prefix of the output FITS files. Default is name
    of the source MATLAB file without '.mat'.
    fitsid_width is width of strings indicating ID of FITS files. Default is 4.
    fitsid_start is the starting number of FITS ID. Default is 1.
    """
    mdict = matlab.loadmat(matlab_name)
    if array_name is None:
        for val in mdict.values():
            if np.size(val)>0 and np.size(np.shape(val))>=2:
                marray = val
    else:
        marray = mdict[array_name]
    if subarray_rng is not None:
        farray = marray[subarray_rng]
    else:
        farray = marray
    NF = np.int(np.prod(np.shape(farray)[2:]))
    NY,NX = np.shape(farray)[0:2]
    farray = farray.reshape((NY,NX,NF))
    if fits_prefix is None:
        fits_prefix = os.path.splitext(matlab_name)[0]
    fitsids = map(lambda n: '0'*(fitsid_width-len(str(n)))+str(n),\
        range(fitsid_start,fitsid_start+NF))
    for f in fitsids:
        fitsname = fits_prefix+'.'+f+'.fits'
        if os.path.isfile(fitsname):
            print str(np.int(f))+'/'+str(NF)+': '+fitsname + ' already exists.'
        else:
            hdu = fits.PrimaryHDU(farray[:,:,np.int(f)-fitsid_start])
            hdu.writeto(fitsname)
            print str(np.int(f))+'/'+str(NF)+': '+fitsname + ' saved.'
