"""This module provides functions that detect interesting features within
a given image.
"""
import numpy as np
import tempfile
import shutil
from astropy.io import fits
from os import path
import os
import subprocess
SEX='/usr/local/bin/sex'
SEX_DEFAULT_SEX="""# Default configuration file for SExtractor 2.5.0
# EB 2006-07-14
#
 
#-------------------------------- Catalog ------------------------------------
 
CATALOG_NAME     catalog        # name of the output catalog
CATALOG_TYPE     ASCII          # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,
                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
PARAMETERS_NAME  default.param  # name of the file containing catalog contents
 
#------------------------------- Extraction ----------------------------------
 
DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)
DETECT_MINAREA   5              # minimum number of pixels above threshold
DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
ANALYSIS_THRESH  1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
 
FILTER           Y              # apply filter for detection (Y or N)?
FILTER_NAME      default.conv   # name of the file containing the filter
 
DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds
DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending
 
CLEAN            Y              # Clean spurious detections? (Y or N)?
CLEAN_PARAM      1.0            # Cleaning efficiency
 
MASK_TYPE        CORRECT        # type of detection MASKing: can be one of
                                # NONE, BLANK or CORRECT

#------------------------------ Photometry -----------------------------------
 
PHOT_APERTURES   5              # MAG_APER aperture diameter(s) in pixels
PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>
PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,
                                # <min_radius>
 
SATUR_LEVEL      50000.0        # level (in ADUs) at which arises saturation
 
MAG_ZEROPOINT    0.0            # magnitude zero-point
MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)
GAIN             0.0            # detector gain in e-/ADU
PIXEL_SCALE      1.0            # size of pixel in arcsec (0=use FITS WCS info)
 
#------------------------- Star/Galaxy Separation ----------------------------
 
SEEING_FWHM      1.2            # stellar FWHM in arcsec
STARNNW_NAME     default.nnw    # Neural-Network_Weight table filename
 
#------------------------------ Background -----------------------------------
 
BACK_SIZE        64             # Background mesh: <size> or <width>,<height>
BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>
# BACK_TYPE        MANUAL
# BACK_VALUE       0.0
BACKPHOTO_TYPE   GLOBAL         # can be GLOBAL or LOCAL
 
#------------------------------ Check Image ----------------------------------
 
CHECKIMAGE_TYPE  APERTURES           # can be NONE, BACKGROUND, BACKGROUND_RMS,
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                # or APERTURES
CHECKIMAGE_NAME  /dev/shm/check.fits     # Filename for the check-image
 
#--------------------- Memory (change with caution!) -------------------------
 
MEMORY_OBJSTACK  3000           # number of objects in stack
MEMORY_PIXSTACK  300000         # number of pixels in stack
MEMORY_BUFSIZE   1024           # number of lines in buffer
 
#----------------------------- Miscellaneous ---------------------------------
 
VERBOSE_TYPE     NORMAL         # can be QUIET, NORMAL or FULL
WRITE_XML        N              # Write XML file (Y/N)?
XML_NAME         sex.xml        # Filename for XML output
"""
SEX_DEFAULT_PARAM="""FLUX_ISO
FLUXERR_ISO
X_IMAGE
Y_IMAGE
X2_IMAGE
Y2_IMAGE
XY_IMAGE
FLAGS
"""
SEX_DEFAULT_CONV="""CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1
"""
def sex(im):
    """Wrapper of sextractor.
    """
    workpath = tempfile.mkdtemp()
    hdulst = fits.HDUList(fits.PrimaryHDU(data=im))
    im_fits = path.join(workpath,'image.fits')
    hdulst.writeto(im_fits)
    with open(path.join(workpath,'default.sex'),'w') as f:
        f.write(SEX_DEFAULT_SEX)
    with open(path.join(workpath,'default.param'),'w') as f:
        f.write(SEX_DEFAULT_PARAM)
    with open(path.join(workpath,'default.conv'),'w') as f:
        f.write(SEX_DEFAULT_CONV)
    curpath = path.abspath(path.curdir)
    os.chdir(workpath)
    subprocess.call([SEX, "image.fits"])
    os.chdir(curpath)
    params = SEX_DEFAULT_PARAM.split()
    try:
        with open(path.join(workpath,'catalog'),'r') as f:
            cat = f.readlines()
    except:
        cat = [[] for i in range(len(params))]
    shutil.rmtree(workpath)
    cat = zip(*tuple(map(lambda s:s.split(),cat)))
    print cat
    cdict = {}
    for i in range(len(params)):
        param = params[i]
        if param == 'FLAGS':
            cdict[param] = np.int64(cat[i])
        else:
            cdict[param] = np.double(cat[i])
    srclist = []
    try:
        for i in range(len(cdict[params[0]])):
            src = {}
            for key in cdict:
                src[key] = cdict[key][i]
            srclist.append(src)
    except:
        print 'No source detected.'
    return srclist,cdict
