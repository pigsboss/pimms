"""Optical system modelling and analysis functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from pymath.common import norm
import pymath.quaternion as quat

class SyntheticAperture(object):
    """Simplified optical model for synthetic-aperture system.
"""
    def __init__(self, bext=None, dext=None):
        pass

class BeamCompressor(object):
    """Simplified optical model for beam compressor system.

Simplified beam compressor model:
The beam compressor model is a combination of a primary concave parabolic mirror and a
secondary convex parabolic mirror that share the same focus.
The focus lies at the origin of the coordinate system.
Equation of the primary mirror:
z = -f + (x^2 + y^2)/4/f,
where f is the focal length of the primary mirror.
Given the f-number of the primary mirror fn, its diameter
d = f / fn,
and the diameter of the secondary mirror
d' = d/r,
where r is the compression ratio.
So the focal length of the secondary mirror
f' = f/r,
thus its equation is
z = -f/r + (x^2 + y^2)*r/4/f.
Finally there is a hole at the center of the primary mirror with the same diameter as the
secondary mirror.

"""
    def __init__(self, diameter=1., ratio=2., fnumber=10., detector=-10., resolution=129):
        """Construct a beam compressor model.

diameter   - diameter of primary mirror, in meter.
ratio      - beam compression ratio, i.e., primary_diameter : secondary_diameter.
fnumber    - f-number of primary mirror.
detector   - z-coordinate of beam amplitude detector.
resolution - spatial resolution of detector in term of pixels.
"""
        self.d=diameter
        self.f=diameter*fnumber
        self.r=ratio
        self.dpos=detector
        self.dpts=resolution

    def __call__(self, phi, theta, wavelength=5e-7):
        """Compute amplitude sampled by predefined detector.

Source model:
Monochromatic point source at infinity, with azimuthal angle (from x-axis) and polar angle
(from z-axis) being phi and theta.

Input arguments;
phi and theta  - angular coordiante of point source.
wavelength     - wavelength of point source, in meter.

Returns:
amp and pha - detected amplitude and phase angles.
"""
        xgv = ((np.arange(self.dpts)+.5)/self.dpts - .5)*self.d
        ygv = ((np.arange(self.dpts)+.5)/self.dpts - .5)*self.d
        x0, y0 = np.meshgrid(xgv, ygv) # meshgrid on primary mirror surface
        rs  = (x0**2.+y0**2.)**.5
        z0  = (x0**2.+y0**2.)/4./self.f - self.f               # z-coordinates of primary mirror
        z1m = self.d**2./self.r/self.f/16. - self.f/self.r
        x01 = x0+(z1m-z0)*np.tan(theta)*np.cos(phi)
        y01 = y0+(z1m-z0)*np.tan(theta)*np.sin(phi)
        # find normal direction of primary mirror
        # surface equation: f(x, y, z) = 0,
        # normal vector: (fx(x0, y0, z0), fy(x0, y0, z0), fz(x0, y0, z0)).
        xn = x0/2./self.f
        yn = y0/2./self.f
        zn = -1.
        phin, thetan, _ = quat.xyz2ptr(xn, yn, zn) # Euler's angles of normal vector
        qz2n = quat.multiply(
            quat.from_angles(phin, thetan),
            quat.conjugate(quat.from_angles(0., np.pi/2.))) # Quaternion that rotates z-axis to normal vector
        # incident vector:
        xi = -np.sin(theta)*np.cos(phi)
        yi = -np.sin(theta)*np.sin(phi)
        zi = -np.cos(theta)
        ri = quat.rotate(quat.conjugate(qz2n), [xi, yi, zi]) # incident vector in local csys, where normal vector being the z-axis
        rr = quat.rotate(qz2n, [ri[0], ri[1], -ri[2]]) # reflected vector in lab csys.
        
        # find arrival coordinates on secondary mirror (x1, y1, z1)
        # parametric equation of light ray from primary mirror to secondary mirror:
        # x = x0 + rr[0]*t
        # y = y0 + rr[1]*t
        # z = z0 + rr[2]*t
        # and z = (x^2+y^2)*r/4/f - f/r (secondary mirror surface equation)
        # A*t^2 + B*t + C = 0, provided rr[2]>0, we have t>0.
        A = (rr[0]**2. + rr[1]**2.)*self.r/self.f/4.
        B = (rr[0]*x0  + rr[1]*y0 )*self.r/self.f/2. - rr[2]
        C = (x0**2.    + y0**2.   )*self.r/self.f/4. - self.f/self.r - z0
        D = B**2. - 4.*A*C
        t = np.zeros_like(x0)
        m1 = np.isclose(A, 0.)
        m2 = np.isclose(C, 0.)
        t[~m1] = (-B[~m1] - (B[~m1]**2. - 4.*A[~m1]*C[~m1])**.5)/A[~m1]/2.
        t[m1&(~m2)] = -B[m1&(~m2)]/C[m1&(~m2)]
        x1 = x0+rr[0]*t
        y1 = y0+rr[1]*t
        z1 = z0+rr[2]*t
        
        # normal vector on secondary mirror:
        xn = x1*self.r/2./self.f
        yn = y1*self.r/2./self.f
        zn = -1.
        phin, thetan, _ = quat.xyz2ptr(xn, yn, zn)
        qz2n = quat.multiply(
            quat.from_angles(phin, thetan),
            quat.conjugate(quat.from_angles(0., np.pi/2.)))
        ri = quat.rotate(quat.conjugate(qz2n), rr)
        rr = quat.rotate(qz2n, [ri[0], ri[1], -ri[2]])
        # find coordinates on exit aperture
        z0m = (self.d/self.r)**2./16./self.f - self.f
        t   = (z0m - z1) / rr[2]
        x2  = x1+rr[0]*t
        y2  = y1+rr[1]*t
        # find transmittance map
        tran = np.double(
            (rs>(self.d/self.r*.5)) &
            (rs<=(self.d*.5)) &
            ((x01**2.+y01**2.)**.5>(self.d*.5/self.r)) &
            ((x2**2.+y2**2.)**.5<=(self.d*.5/self.r)))
        # find arrival coordinates on detector plane
        t  = (self.dpos - z1) / rr[2]
        x3 = x1+rr[0]*t
        y3 = y1+rr[1]*t
        # total OP
        t   = xi*x0+yi*y0+zi*z0
        op0 = t
        op1 = ((x1-x0)**2. + (y1-y0)**2. + (z1-z0)**2.)**.5
        op2 = ((x3-x1)**2. + (y3-y1)**2. + (self.dpos-z1)**2.)**.5
        opd = op0+op1+op2 - np.mean(op0+op1+op2)
        re  = tran*np.cos(2.*np.pi*opd/wavelength)
        im  = tran*np.sin(2.*np.pi*opd/wavelength)
        xid = np.int64((x3+self.d/2.)/self.d*self.dpts)
        yid = np.int64((y3+self.d/2.)/self.d*self.dpts)
        pid = yid*self.dpts+xid
        remap = np.bincount(pid.ravel(), weights=re.ravel(), minlength=self.dpts*self.dpts)
        immap = np.bincount(pid.ravel(), weights=im.ravel(), minlength=self.dpts*self.dpts)
        amp = np.reshape((remap**2.+immap**2.)**.5, (self.dpts, self.dpts))
        pha = np.ma.masked_array(np.reshape(np.arctan2(immap, remap), (self.dpts, self.dpts)), mask=~np.bool_(amp>0))
        return amp, pha

def parabolic_reflector_experiment(source, detector, resolution=129, nside=32768, diameter=1., fnumber=10., wavelength=5e-7):
    """Simple parabolic reflector experiment.
Reflector model:
The focus of the parabolic reflector is at the origin of the coordinate system.
The axis of symmetry is along z-axis and the reflector converges lights along -z direction.
Equation of the paraboloid:
z = -f + (x^2 + y^2)/4/f,
where f is the focal length of the paraboloid.

Source model:
Monochromatic point source, emitting scalar spherical waves.
wave function: 
U(x, y, z, t) = 1/r*exp(i*(omega*t-k*r)),
where omega = 2*pi*c/lambda and k = 2*pi/lambda.

source     - (x,y,z) coordinate of source with unit luminosity.
detector   - position and size of acquisition detector, in (x_size, y_size, z_center).
resolution - number of pixels along each axis of the detector.
nside      - nside of equi-areal samplings pixels.
diameter   - diameter of the reflector, in meters.
fnumber    - f number of the reflector, i.e., focal_length / diameter.
wavelength - wavelength of the emission light, in meters.

Returns:
complex amplitude sampled by detector at t=0.
"""
    f = diameter * fnumber
    # equi-areal sample rays from the source
    npixs = hp.nside2npix(nside) // 768
    tx,ty,tz = hp.pix2vec(nside, np.arange(npixs)+767*npixs)
    a = tx**2.+ty**2.
    b = 2.*(tx*tz*source[0]-tx*tx*source[2]+ty*tz*source[1]-ty*ty*source[2]) - 4.*tz*tz*f
    c = (tz*source[0]-tx*source[2])**2. + (tz*source[1]-ty*source[2])**2. - 4.*(tz*f)**2.
    # intersections
    zi = (-b-(b**2.-4.*a*c)**.5) / 2. / a
    xi = (zi-source[2])*tx/tz+source[0]
    yi = (zi-source[2])*ty/tz+source[1]
    collected = ((xi**2.+yi**2.)**.5 < (diameter*.5))

    # normal vector direction, in Euler angles
    theta = np.pi/2. - np.arctan((xi[collected]**2.+yi[collected]**2.)**.5/2./f)
    phi   = np.arctan2(yi[collected], xi[collected]) + np.pi
    # normal vector, from intersection
    xn = np.cos(theta)*np.cos(phi)
    yn = np.cos(theta)*np.sin(phi)
    zn = np.sin(theta)
    # source vector, from intersection
    xs = source[0]-xi[collected]
    ys = source[1]-yi[collected]
    zs = source[2]-zi[collected]
    alpha = np.arcsin(norm(np.cross([xs,ys,zs],[xn,yn,zn],axisa=0,axisb=0,axisc=0)) / norm([xs,ys,zs]) / norm([xn,yn,zn]))
    # reflected vector, from intersection
    xr,yr,zr = 2.*np.cos(alpha)*np.double([xn,yn,zn])*norm([xs,ys,zs]) - np.double([xs,ys,zs])
    ## assert np.allclose(np.arcsin(norm(np.cross([xn,yn,zn],[xr,yr,zr],axisa=0,axisb=0,axisc=0))/norm([xn,yn,zn])/norm([xr,yr,zr])),alpha)

    # intersection with detector plane
    xsize,ysize,zd = detector
    k = (zd-zi[collected]) / zr
    xd = xr*k+xi[collected]
    yd = yr*k+yi[collected]
    # registration
    detected = (np.abs(xd)<xsize*.5) & (np.abs(yd)<ysize*.5)
    xid = np.int64((xd[detected]+xsize/2.) / xsize * resolution)
    yid = np.int64((yd[detected]+ysize/2.) / ysize * resolution)
    pid = yid*resolution+xid
    counts = np.bincount(pid, minlength=resolution*resolution)
    cmap = np.copy(counts.reshape((resolution,resolution)))
    op   = norm([xs,ys,zs])+norm([xd-xi[collected],yd-yi[collected],zd-zi[collected]])
    opd  = op[detected] - np.min(op[detected])
    re = np.cos(2.*np.pi*opd/wavelength) / op[detected]
    im = np.sin(2.*np.pi*opd/wavelength) / op[detected]
    re_mean = np.bincount(pid, weights=re, minlength=resolution*resolution) / np.clip(counts, 1., None)
    im_mean = np.bincount(pid, weights=im, minlength=resolution*resolution) / np.clip(counts, 1., None)
    phase = np.arctan2(im_mean, re_mean)
    amp   = (re_mean**2.+im_mean**2.)**.5
    pmap = np.copy(phase.reshape((resolution,resolution)))
    amap = np.copy(amp.reshape((resolution,resolution)))
    return amap, pmap
    
