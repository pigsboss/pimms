"""This module is used to produce artificial objects from definitive data
sources such as catalogs, spectral models, power spectra, etc. Synthetic
objects are more descriptive compared with their sources. For example, maps,
light curves, as well as energy spectra are such descriptive objects.
This synthetic process is the inverse of catalog compilation or spectral
fitting in data analysis.
"""
import numpy as np
import pymath.sphere as sp
from pymath.common import *
import pymath.quaternion as quaternion
import pymath.functions as functions
import healpy as hp

POINT_SOURCE_SIGMA = 1.0/60/180*np.pi # minimum sigma of extended source, in rad.

def crab_nebula_flux(E):
    """Flux of Crab nebula at given energy (>=20keV).
    Description
    F = crab_nebula_flux(E)
    E is energy in keV.
    F is photon flux in photons per second per cm^2 per keV.
    Reference:
    1. Jourdain, E. and Roques, J. P.,  The high-energy emission of the Crab
    nebula from 20keV to 6 MeV with INTEGRAL/SPI. ApJ, 2009.
    2. Massaro, E., et al., Fine phase resolved spectroscopy of the X-ray
    emission of the Crab pulsar (PSR B0531+21) observed with BeppoSAX.
    """
    a = 1.79
    b = 0.134
    A = 3.87 # pts / cm^2 / s / keV
    E0 = 20.0
    return A * E**(-(a + b*np.log10(E/E0)))

class source(object):
    '''The default source is a crab-like point source at (0,0).
    '''
    def __init__(self,location=(0.0,0.0),\
        intensity=1.0,\
        spectrum=crab_nebula_flux,\
        extension=functions.sphere_gauss(0.0,0.0,POINT_SOURCE_SIGMA),\
        point=True,
        space='euclidean'):
        '''Keyword arguments:
        location is a tuple, either in (x,y) or in (longitude,latitude,roll_angle).
        travelling source is not implemented yet.
        intensity is time-varying intensity of the source, in counts/cm^2/s.
        spectrum is energy spectrum.
        extension is normalized spatial extension of the source.
        space is either 'euclidean' or 'spherical'.
        '''
        args = []
        self.location = location
        # spatial-distribution, unit: depends on space of location coordinates.
        if space == 'euclidean':
            self.extension = functions.function2d(extension)
            self.local_coordinates = lambda x,y:(x-location[0],y-location[1])
            self.distribution = functions.function2d(lambda x,y:extension(*tuple(\
                self.local_coordinates(x,y))))
            args += ['x','y']
        elif space == 'spherical':
            self.extension = functions.sphere2d(extension)
            self.quat = quaternion.from_angles(*tuple(location))
            self.local_coordinates = lambda x,y:xyz2ptr(*tuple(quaternion.rotate(\
                quaternion.conjugate(self.quat),np.array(ptr2xyz(x,y)))))[0:2]
            self.distribution = functions.sphere2d(lambda x,y:extension(*tuple(\
                self.local_coordinates(x,y))))
            args += ['longitude','latitude']
        else:
            raise StandardError('Unrecognized space.')
        args += ['time','energy']
        self.args = args
        # time-varying intensity, unit: counts/cm^2/s
        self.intensity = functions.function(intensity)
        # energy spectrum, unit: keV^-1
        self.spectrum = functions.function(spectrum)
        self.space = space
        self.location_unit = 'rad'
        self.time_unit = 's'
        self.energy_unit = 'keV'
        self.area_unit = 'sr'
        self.__function__ = functions.multiply((self.distribution,2),\
            (self.intensity,1),(self.spectrum,1))[0]
        self.point = point
    def counts(self,fov=None,exposure=None,energy_range=None):
        '''Specify FoV, duration of exposure, or energy range to integrate
        the spatial distribution, time-varying intensity, or energy spectrum
        respectively.
        fov can be specified either in numeric upper limits and low limits or
        'sup' (support).
        '''
        unit = 'counts/cm^2'
        args = []
        if fov is None:
            unit = unit+'/'+self.area_unit
            fov_func = (self.distribution,2)
            args += self.args[0:2]
        elif fov.lower()[0:3] == 'sup':
            fov_func = (1.0,0)
        else:
            fov_func = (self.distribution.integral(fov[0],fov[1]),0)
        if exposure is None:
            unit = unit + '/' + self.time_unit
            exposure_func = (self.intensity,1)
            args.append('time')
        else:
            exposure_func = (self.intensity.integral(exposure),0)
        if energy_range is None:
            unit = unit + '/' + self.energy_unit
            energy_func = (self.spectrum,1)
            args.append('energy')
        else:
            energy_func = (self.spectrum.integral(energy_range),0)
        return functions.multiply(fov_func,exposure_func,energy_func),unit,args
    def __call__(self,*args):
        '''Arguments are given in (coordinates, time, energy), e.g.,
        (x, y, time, energy) or (phi, theta, time, energy).
        '''
        return self.inner_function(*args)

class world(object):
    def __init__(self,space='euclidean',csys='cartesian'):
        '''space is either 'euclidean' or 'spherical'.
        csys is name of coordinate system.
        available coordinate systems are: ecliptic, galactic, equatorial,
        spherical, and cartesian.
        '''
        self.sources = []
        self.space = space
        self.csys = csys
        if csys.lower()[:2] in ['ec','ga','eq','sp']:
            self.coordinates = ['lon','lat']
            self.distance = lambda a,b:sp.distance(a,b)
            sefl.angle = lambda a,b:sp.angle(a,b)
        elif csys.lower()[:2]=='ca':
            self.coordinates = ['x','y']
            self.distance = \
                lambda a,b:np.sqrt(np.sum(np.power(np.subtract(a.T,b.T),\
                    2.0).T,axis=0))
            self.angle = lambda a,b:np.arctan2(b[1]-a[1],b[0]-a[0])
    def append(self,s):
        '''Append a new source to the sources list of the current world object.
        '''
        if s.space == self.space:
            self.sources.append(s)
        else:
            print 'The source is in a different space. Ignored.'
    def __call__(self,*args):
        '''Arguments are given in (coordinates, time, energy), e.g.,
        (x, y, time, energy) or (phi, theta, time, energy).
        '''
        if len(self.sources)==0:
            return None
        elif len(self.sources)==1:
            return self.sources[0](*args)
        else:
            return reduce(lambda a,b:a+b,map(lambda s:s(*args),self.sources))
    def snapshot(self,time_interval,\
        extent=((-0.1,0.1),(-0.1,0.1)),\
        energy_range=(20.0,250.0),\
        resolution=None,npix=None):
        '''Take a snapshot of the current world object.
        A snapshot is an image of the object as if observed with an ideal
        imaging system.
        time_interval is a 2-element tuple. The first element is the moment when
        the exposure begins. The last element is the moment when the exposure
        ends.
        extent is used to specify the extent of the snapshot if the world coordinate
        system is euclidean. The extend is specified in ((xmin, xmax), (ymin, ymax)).
        energy_range is a tuple in (e_min, e_max) that specifies the energy range of
        the snapshot.
        Either resolution or npix is given to specify the resolution or the number of
        pixels of the snapshot. If both are given, npix will be overrided.
        '''
        if self.space == 'spherical':
            if resolution is None:
                if npix is None:
                    nside = 1024
                else:
                    nside = hp.npix2nside(npix)
            else:
                nside = 2**np.int64(np.ceil(np.log2(1024.0*\
                    hp.nside2resol(1024)/resolution)))
            npix = hp.nside2npix(nside)
            hpx = np.zeros(npix)
            for s in self.sources:
                if s.point:
                    phi,theta = s.location[0],s.location[1]
                    pix = hp.ang2pix(nside,np.pi/2.0 - theta,phi,nest=False)
                    hpx[pix] += s.counts(exposure=time_interval,\
                        fov='support',energy_range=energy_range)[0][0]
                else:
                    theta,phi = hp.pix2ang(nside,range(0,npix))
                    hpx = hpx + s.counts(exposure=time_interval,\
                        energy_range=energy_range)[0][0](phi,np.pi/2.0-theta)*\
                        hp.nside2resol(nside)**2.0
            return hpx
        elif self.space == 'euclidean':
            if resolution is None:
                if npix is None:
                    npix = 1024
                resolution = np.double(extent[0][1] - extent[0][0]) / npix
            else:
                npix = np.int64(np.double(extent[0][1] - extent[0][0]) /\
                    resolution)
            nx = np.int64(np.double(extent[0][1] - extent[0][0]) /\
                resolution)
            ny = np.int64(np.double(extent[1][1] - extent[1][0]) /\
                resolution)
            im = np.zeros((ny,nx))
            for s in self.sources:
                if s.point:
                    x,y = s.location[0],s.location[1]
                    xi = np.int64((x-extent[0][0])/resolution)
                    yi = np.int64((y-extent[1][0])/resolution)
                    im[yi,xi] += s.counts(exposure=time_interval,\
                        fov='support',energy_range=energy_range)[0][0]
                else:
                    xi,yi = np.meshgrid(np.arange(0.0,nx),np.arange(0.0,ny))
                    x = xi*resolution+extent[0][0]
                    y = yi*resolution+extent[1][0]
                    im = im + s.counts(exposure=time_interval,\
                        energy_range=energy_range)[0][0](x,y)*\
                        resolution**2.0
            return im
