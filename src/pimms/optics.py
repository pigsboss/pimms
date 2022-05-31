"""Optical system modelling and analysis functions.
"""
import numpy as np
import networkx as nx
import numexpr as ne
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import healpy as hp
import copy
import pymath.quaternion as quat
import pymath.linsolvers as lins
from time import time
from mayavi import mlab
from matplotlib import rcParams
from pymath.common import norm

hc = 1.98644586e-25 # speed of light * planck constant, in m3/kg/s2.
dm = 1e-5           # mirror safe thickness to separate top from bottom, in m.

# super photon properties data type:
sptype = np.dtype([
    ('weight',     'f4'),
    ('position',   'f8', 3),
    ('direction',  'f8', 3),
    ('wavelength', 'f8'),
    ('phase',      'f8')])
# mirror sequence data type:
mstype = 'int16'

def gimbal(u, phi, theta):
    """Perturbs unit vector u around an axis a by angle theta, while
    the angle between a and its co-vector v is phi.
    In a certain coordinate system where u=(1,0,0) and v=(0,0,1),
    a=(cos(phi), sin(phi), 0).

    Arguments:
    u      - 3D unit vectors, in (N, 3)
    phi    - azimuth angles, or angles between a and v, in (N,)
    theta  - elevation angles, or angles around a, in (N,)

    Returns:
    u_new  - Perturbed unit vector, in (N, 3)
    """
    beta = np.arctan2(u[:,1], u[:,0])
    u_xy = u[:,0]**2. + u[:,1]**2.
    cos_alpha = u_xy**.5
    sin_alpha = (1. - u_xy)**.5
    cos_theta = np.cos(theta/2.)
    sin_theta = np.sin(theta/2.)
    q = np.double([
        cos_theta,
        -sin_theta*sin_alpha*np.cos(beta),
        -sin_theta*sin_alpha*np.sin(beta),
        sin_theta*cos_alpha])
    u_new = quat.rotate(q, u.transpose())
    return u_new.transpose()

class LightSource(object):
    def __init__(self, location=(0., 0., np.inf), intensity=1e-10, wavelength=5e-7):
        """Create simple light source model.

        Arguments:
        location   - location of the light source in lab coordinate system,
                     (phi, theta, rho).
                     phi is the angle between the source vector and x-axis on xOy plane.
                     theta is the angle between the source vector and z-axis, i.e., co-latitude.
                     rho is distance between the source and the origin, in meters.
                     Default: infinity towards +z.
        intensity  - integral flux at the origin of the lab coordinate system, in W/m2.
        wavelength - wavelength of the light source, in m.
        """
        self.phi,self.theta,self.rho = location
        self.intensity  = intensity
        self.wavelength = wavelength
        self.energy     = hc / self.wavelength
        self.direction  = np.double([
            np.sin(self.theta)*np.cos(self.phi),
            np.sin(self.theta)*np.sin(self.phi),
            np.cos(self.theta)]) # unit vector from the origin to the source
        self.num_crosshairs = 8
    def __call__(self, list_of_apertures, num_super_photons, dt, sampling='dizzle'):
        """Shed super photons onto mirrors.

        list_of_apertures - list of apertures of optical system
        num_super_photons - number of super photons per aperture
        dt                - atomic time duration, in second

        An aperture is the hole next to the object glass of the telescope, through which
        the light and image of the object comes into the tube and thence it is carried to
        the eye. (1707, Glossographia Anglicana Nove.)

        To trace light rays from the light source end to the detector end of the optical
        system a virtual aperture is placed in front (the object side) of the primary
        mirror of each photon collecting telescope. Such a virtual aperture is implemented
        with a virtual plane mirror with necessary properties such as attitude quaternion,
        position and outside diameter.

        A super photon is an atomic envelope of energy emitted from a light source during
        an indivisible period of time.
        Properties of a super photon:
        weight    - a measure of energy encapsulated
        position  - a 3D coordinate in lab csys
        direction - a unit vector
        phase     - phase angle at starting point

        Returns:
        sp_start  - super photons passing through an equi-phase surface between the exit pupil
                    of the light source and the apertures.
        sp_stop   - super photons arriving at the apertures
        """
        #
        # find super photons at equi-phase surface sq_start.
        if np.isinf(self.rho):
            # plane wave
            op = [] # length of optical path from the equi-phase plane to the nearest point on the aperture
            area   = 0. # total area of exit pupil
            for aperture in list_of_apertures:
                norm_aperture = quat.rotate(aperture.q, [0., 0., 1.]) # normal vector of aperture
                sin_alpha = np.linalg.norm(np.cross(self.direction, norm_aperture, axis=0))
                # displacement from the center of the aperture to the equi-phase plane passing the origin, i.e.,
                # (x,y,z) dot self.direction = 0, along the direction towards the source.
                t = -aperture.p[0]*self.direction[0]-aperture.p[1]*self.direction[1]-aperture.p[2]*self.direction[2]
                op.append(t-aperture.d_out*sin_alpha*.5)
                area += np.pi*(aperture.d_out**2.-aperture.d_in**2.)/4.*sin_alpha
            opm = np.min(op) # minimum optical path
            u = []
            for aperture in list_of_apertures:
                if sampling.lower()=='random':
                    rho = (((aperture.d_out**2.-aperture.d_in**2.)*np.random.rand(num_super_photons)+aperture.d_in**2.)**.5)/2.
                    phi = 2.*np.pi*np.random.rand(num_super_photons)
                elif sampling.lower()=='uniform':
                    N = int((num_super_photons/np.pi/(aperture.d_out**2.-aperture.d_in**2.)*4.*aperture.d_out**2.)**.5)
                    xgv = ((np.arange(N)+.5)/N-.5)*aperture.d_out
                    ygv = ((np.arange(N)+.5)/N-.5)*aperture.d_out
                    x,y = np.meshgrid(xgv,ygv)
                    rho = (x.ravel()**2.+y.ravel()**2.)**.5
                    phi = np.arctan2(y.ravel(),x.ravel())
                    sel = (rho>=aperture.d_in*.5) & (rho<=aperture.d_out*.5)
                    rho = rho[sel]
                    phi = phi[sel]
                elif sampling.lower()=='dizzle':
                    N = int((num_super_photons/np.pi/(aperture.d_out**2.-aperture.d_in**2.)*4.*aperture.d_out**2.)**.5)
                    xgv = ((np.arange(N)+.5)/N-.5)*aperture.d_out+(np.random.rand(1)-.5)*aperture.d_out/N
                    ygv = ((np.arange(N)+.5)/N-.5)*aperture.d_out+(np.random.rand(1)-.5)*aperture.d_out/N
                    x,y = np.meshgrid(xgv,ygv)
                    rho = (x.ravel()**2.+y.ravel()**2.)**.5
                    phi = np.arctan2(y.ravel(),x.ravel())
                    sel = (rho>=aperture.d_in*.5) & (rho<=aperture.d_out*.5)
                    rho = rho[sel]
                    phi = phi[sel]
                elif sampling.lower()=='crosshair':
                    xgv = np.arange(num_super_photons) / num_super_photons # fine grids
                    ygv = (np.arange(self.num_crosshairs)+.5) / self.num_crosshairs # coarse grids
                    x,y = np.meshgrid(xgv,ygv)
                    rho = (((aperture.d_out**2.-aperture.d_in**2.)*np.concatenate((x.ravel(), y.ravel()))+aperture.d_in**2.)**.5)/2.
                    phi = 2.*np.pi*np.concatenate((y.ravel(), x.ravel()))
                # samplings of aperture in lab coordinates
                r = quat.rotate(aperture.q, [rho*np.cos(phi), rho*np.sin(phi), 0.]) + aperture.p
                t = -(r[0]*self.direction[0]+r[1]*self.direction[1]+r[2]*self.direction[2])-opm
                s = np.double([
                    r[0]+self.direction[0]*t,
                    r[1]+self.direction[1]*t,
                    r[2]+self.direction[2]*t])
                u.append(s)
            u = np.concatenate(u, axis=1)
            sp_start = np.zeros((u.shape[1],), dtype=sptype)
            sp_start['weight']    = self.intensity*dt*area/self.energy/u.shape[1]
            sp_start['position']  = u.transpose()
            sp_start['direction'] = -self.direction
        else:
            # spherical wave
            p = np.reshape(quat.ptr2xyz(self.phi, np.pi/2.-self.theta, self.rho), (3,))
            far_field = True
            for aperture in list_of_apertures:
                far_field = far_field & (np.linalg.norm(p - aperture.p.ravel()) > aperture.d_out*.5)
            if far_field:
                u = []
                area = 0.
                for aperture in list_of_apertures:
                    R = np.linalg.norm(p - aperture.p.ravel())
                    r = aperture.d_out*.5
                    norm_aperture = quat.rotate(aperture.q, [0., 0., 1.]) # normal vector of aperture
                    # alpha is the angle between norm vector of aperture and chief ray from the source
                    # chief ray is the ray that passes the center of the aperture
                    sin_alpha = np.linalg.norm(np.cross(self.direction, norm_aperture, axis=0))
                    # beta is the upper limit of the angle between chief ray and the marginal ray
                    # marginal ray is the ray that passes the edge of the aperture
                    if sin_alpha >= (r/R):
                        beta = np.arccos(((R**2.-r**2.)**.5) / R)
                    else:
                        beta = np.arccos((R-r*sin_alpha) / (R**2.+r**2.-2.*R*r*sin_alpha)**.5)
                    if sampling.lower()=='random':
                        rho = R*((1.-(1.-np.cos(beta))*np.random.rand(num_super_photons))**-2. - 1.)**.5
                        phi = 2.*np.pi*np.random.rand(num_super_photons)
                        xs  = rho*np.cos(phi)
                        ys  = rho*np.sin(phi)
                    elif sampling.lower()=='uniform':
                        nside = 2**int(np.log2((num_super_photons/6./(1.-np.cos(beta)))**.5))
                        npix_stop = hp.ang2pix(nside, beta, 0.)
                        npix_half = hp.ang2pix(nside, beta, np.pi)
                        npix_stop = int(npix_stop + 2*(npix_half-npix_stop))
                        xs, ys, _ = np.double(hp.pix2vec(nside, range(npix_stop)))*R
                    elif sampling.lower()=='dizzle':
                        nside = 2**int(np.log2((num_super_photons/6./(1.-np.cos(beta)))**.5))
                        npix_stop = hp.ang2pix(nside, beta, 0.)
                        npix_half = hp.ang2pix(nside, beta, np.pi)
                        npix_stop = int(npix_stop + 2*(npix_half-npix_stop))
                        xs,ys,zs  = np.double(hp.pix2vec(nside, range(npix_stop)))*R
                        dizzle_axis = quat.ptr2xyz(np.pi*2.*np.random.rand(1), 0., 1.)
                        dizzle_dist = np.random.rand(1)*hp.nside2resol(nside)/2.
                        dizzle_quat = np.double([
                            np.cos(dizzle_dist/2.),
                            np.sin(dizzle_dist/2.)*dizzle_axis[0],
                            np.sin(dizzle_dist/2.)*dizzle_axis[1],
                            np.sin(dizzle_dist/2.)*dizzle_axis[2]])
                        xs,ys,zs  = quat.rotate(dizzle_quat, [xs,ys,zs])
                    elif sampling.lower()=='crosshair':
                        xgv = np.arange(num_super_photons) / num_super_photons # fine grids
                        ygv = (np.arange(self.num_crosshairs)+.5) / self.num_crosshairs # coarse grids
                        x,y = np.meshgrid(xgv,ygv)
                        rho = R*((1.-(1.-np.cos(beta))*np.concatenate((x.ravel(),y.ravel())))**-2. - 1.)**.5
                        phi = 2.*np.pi*np.concatenate((y.ravel(),x.ravel()))
                        xs  = rho*np.cos(phi)
                        ys  = rho*np.sin(phi)
                    s = quat.rotate(quat.from_angles(*(quat.xyz2ptr(
                        aperture.p[0]-p[0],
                        aperture.p[1]-p[1],
                        aperture.p[2]-p[2])[:2])), [R,xs,ys])
                    u.append(s)
                    area += 2.*np.pi*(1.-np.cos(beta))*(self.rho**2.)
                u = np.concatenate(u, axis=1)
                sp_start = np.zeros((u.shape[1],), dtype=sptype)
                sp_start['weight']    = self.intensity*dt*area/self.energy/u.shape[1]
                sp_start['direction'] = quat.direction(u).transpose()
            else:
                if sampling.lower()=='random':
                    theta = np.arccos(1.-2.*np.random.rand(num_super_photons))
                    phi   = 2.*np.pi*np.random.rand(num_super_photons)
                elif sampling.lower() in ['crosshair', 'uniform']:
                    nside = 2**int(np.log2((num_super_photons/12.)**.5))
                    npix  = hp.nside2npix(nside)
                    theta, phi = hp.pix2ang(nside, range(npix))
                elif sampling.lower()=='dizzle':
                    nside = 2**int(np.log2((num_super_photons/12.)**.5))
                    npix  = hp.nside2npix(nside)
                    theta, phi = hp.pix2ang(nside, range(npix))
                    dizzle_axis = quat.ptr2xyz(np.pi*2.*np.random.rand(1), 0., 1.)
                    dizzle_dist = np.random.rand(1)*hp.nside2resol(nside)/2.
                    dizzle_quat = np.double([
                        np.cos(dizzle_dist/2.),
                        np.sin(dizzle_dist/2.)*dizzle_axis[0],
                        np.sin(dizzle_dist/2.)*dizzle_axis[1],
                        np.sin(dizzle_dist/2.)*dizzle_axis[2]])
                    xs,ys,zs  = quat.rotate(dizzle_quat, np.double(quat.ptr2xyz(phi, np.pi/2.-theta, 1.)))
                    theta, phi  = quat.xyz2ptr(xs, ys, zs)
                    theta = np.pi/2.-theta
                sp_start = np.zeros((num_super_photons,), dtype=sptype)
                sp_start['weight']    = self.intensity*dt*(4.*np.pi*self.rho**2.)/self.energy/num_super_photons
                sp_start['direction'] = np.double(quat.ptr2xyz(phi, np.pi/2.-theta, 1.)).transpose()
            sp_start['position'] = p
        sp_start['phase'] = 0.
        #
        # find super photons at aperture sp_stop.
        sp_stop = np.empty_like(sp_start)
        sp_stop['weight'][:]    = sp_start['weight']
        sp_stop['position'][:]  = np.nan
        sp_stop['direction'][:] = sp_start['direction']
        sp_dist = np.empty((sp_start.size, ))
        sp_dist[:] = np.inf
        for aperture in list_of_apertures:
            n, t = aperture.intersect(sp_start)
            sp_stop['position'][:] = np.where((t<sp_dist).reshape(-1,1), np.transpose(quat.rotate(aperture.q,n)+aperture.p), sp_stop['position'])
            sp_dist = np.where(t<sp_dist, t, sp_dist)
        miss_all = np.isinf(sp_dist)
        sp_stop['phase'][ miss_all] = np.nan
        sp_stop['phase'][~miss_all] = 2.*np.pi*np.mod(sp_dist[~miss_all], self.wavelength)/self.wavelength
        sp_start['wavelength'][:] = self.wavelength
        sp_stop['wavelength'][:] = self.wavelength
        return sp_start[~miss_all], sp_stop[~miss_all]

class SymmetricQuadricMirror(object):
    """Symmetric quadric mirror model.

    1. Plane mirror
    center of mirror is at origin.
    mirror lies on xOy plane.
    norm vector is along z-axis.
    Equation: f(x,y,z) = z = 0.
    Parameters:
    d_in  - inside diameter
    d_out - outside diameter
    b     - reflectance, in (z+, z-)
    f     - np.inf
    g     - np.inf
    
    2. Parabolic mirror
    focus of mirror is at origin.
    mirror opens upward (+z direction).
    Equation: f(x,y,z) = x^2 + y^2 - 4f z - 4f^2 = 0.
    Parameters:
    d_in  - inside diameter
    d_out - outside diameter
    b     - boundary type, in (z+, z-)
    f     - focal length
    g     - np.inf
    
    3. Hyperbolic mirror
    focus on the same side is at origin.
    mirror opens upward (+z direction).
    Equation: f(x,y,z) = x^2 + y^2 + (a^2-c^2)/a^2 z^2 + 2(a^2-c^2)c/a^2 z - (a^2-c^2)^2/a^2 = 0,
    where a = (f+g)/2 and c = (f-g)/2.
    Parameters:
    d_in  - inside diameter
    d_out - outside diameter
    b     - boundary type, in (z+, z-)
    f     - focal length, displacement from mirror center to the focus of the same side, f>0.
    g     - focal length, displacement from mirror center to the focus of the other side, g<0.

    4. Elliptical mirror
    the near focus is at origin.
    mirror opens upward (+z direction).
    Equation: f(x,y,z) = x^2 + y^2 + (a^2-c^2)/a^2 z^2 + 2(a^2-c^2)c/a^2 z - (a^2-c^2)^2/a^2 = 0,
    where a = (f+g)/2 and c = (f-g)/2.
    Parameters:
    d_in  - inside diameter
    d_out - outside diameter
    b     - boundary type, in (z+, z-)
    f     - focal length, displacement from mirror center to the near focus, f>0.
    g     - focal length, displacement from mirror center to the far focus, g>f>0.
    """
    def __init__(
            self,
            d_in,
            d_out,
            b=(1, 1),
            f=np.inf,
            g=np.inf,
            p=[0., 0., 0.],
            q=[1., 0., 0., 0.],
            name='unnamed',
            is_primary=False,
            is_virtual=False,
            is_entrance=False,
            is_detector=False
    ):
        """Construct symmetric quadric mirror object.

        Arguments:
        d_in  - inside diameter.
        d_out - outside diameter.
        b     - boundary type in (top, bottom) of top (towards the focus, or +z direction) and
                bottom (away from the focus, or -z direction) surface of the mirror:
                +1 - fully reflection;
                 0 - fully absorption;
                -1 - fully transmission.
        f     - focal length, for parabolic and hyperbolic mirrors.
        g     - distance to the other focus, for hyperbolic mirror only.
        p     - position in lab coordinate system.
        q     - attitude quaternion to convert coordinate of mirror's fixed csys to lab csys.
                example: r' = qrq' + p,
                where r is coordinate of mirror's fixed csys, r' is coordinate of lab csys,
                p is position of the mirror in lab csys and q is the attitude quaternion.
        name  - human readable name assigned to the mirror.
        is_primary, is_virtual, is_entrance and is_detector are reserved switches for
        OpticalSystem object.
        """
        self.d_in     = d_in
        self.d_out    = d_out
        self.boundary = b
        self.f        = f
        self.g        = g
        self.p        = np.double(p).reshape((3,1))
        self.q        = np.double(q).reshape((4,1))
        self.name     = name
        self.is_primary  = bool(is_primary)
        self.is_virtual  = bool(is_virtual)
        self.is_entrance = bool(is_entrance)
        self.is_detector = bool(is_detector)
        if np.isinf(self.f):
            # plane mirror
            self.coef = {
                'x2':0.,
                'y2':0.,
                'z2':0.,
                'xy':0.,
                'xz':0.,
                'yz':0.,
                'x' :0.,
                'y' :0.,
                'z' :1.,
                'c' :0.
            }
        elif np.isinf(self.g):
            # parabolic mirror
            self.coef = {
                'x2':1.,
                'y2':1.,
                'z2':0.,
                'xy':0.,
                'xz':0.,
                'yz':0.,
                'x' :0.,
                'y' :0.,
                'z' :-4.*self.f,
                'c' :-4.*self.f**2.
            }
        elif self.g<0.:
            # hyperbolic mirror
            a = (self.f+self.g)*.5
            c = (self.f-self.g)*.5
            self.coef = {
                'x2':1.,
                'y2':1.,
                'z2':(a**2.-c**2.)/a**2.,
                'xy':0.,
                'xz':0.,
                'yz':0.,
                'x' :0.,
                'y' :0.,
                'z' :2.*(a**2.-c**2.)*c/a**2.,
                'c' :-(a**2.-c**2.)**2./a**2.
            }
        elif self.g>0.:
            # elliptical mirror
            a = (self.f+self.g)*.5
            c = (self.f-self.g)*.5
            self.coef = {
                'x2':1.,
                'y2':1.,
                'z2':(a**2.-c**2.)/a**2.,
                'xy':0.,
                'xz':0.,
                'yz':0.,
                'x' :0.,
                'y' :0.,
                'z' :2.*(a**2.-c**2.)*c/a**2.,
                'c' :-(a**2.-c**2.)**2./a**2.
            }

    def draw(self, nside=32, axes=None, return_only=False, **kwargs):
        """Draw triangulated 3D surface plot of the mirror in lab csys.

        Arguments:
        nside - nside of healpix sampling points over the surface of the mirror.
        axes  - predefined matplotlib axes object.
        return_only - instead of plotting surface, return the triangles only.
        """
        # healpix sampling points
        # use 1/12 shpere to sample the surface of the mirror
        # 1 - cos(theta) = 2*n/N, where theta is the co-latitude,
        # n is number of pixels (ring scheme) and N is number of all pixels.
        # n=N/12, so cos(theta)=5/6 and sin(theta)=(11/36)^0.5=0.5527708
        theta_out = np.arcsin(.5527708)
        npix_stop = hp.ang2pix(nside, theta_out, 0.)
        npix_half = hp.ang2pix(nside, theta_out, np.pi)
        npix_stop = int(npix_stop + 2*(npix_half-npix_stop))
        xs, ys, _ = np.double(hp.pix2vec(nside, range(npix_stop)))*self.d_out
        rs = (xs**2.+ys**2.)**.5
        if np.isinf(self.f):
            zs = np.zeros_like(xs)
        elif np.isinf(self.g):
            zs = -(self.coef['x2']*xs**2. + self.coef['y2']*ys**2. + self.coef['c']) / self.coef['z']
        else:
            a = self.coef['z2']
            b = self.coef['z']
            c = self.coef['x2']*xs**2. + self.coef['y2']*ys**2. + self.coef['c']
            zs = (-b-(b**2.-4.*a*c)**.5)/2./a
        # triangulation
        tri = mtri.Triangulation(xs, ys)
        rc = (xs[tri.triangles].mean(axis=-1)**2.+ys[tri.triangles].mean(axis=-1)**2.)**.5
        # mask out triangles out of boundaries (d_out & d_in)
        mask = np.where(np.logical_or(rc>(self.d_out*.5), rc<(self.d_in*.5)), 1, 0)
        tri.set_mask(mask)
        # convert mirror fixed coordinates to lab coordinate system
        R = quat.rotate(self.q, [xs,ys,zs])
        X = R[0]+self.p[0]
        Y = R[1]+self.p[1]
        Z = R[2]+self.p[2]
        tri.x = X
        tri.y = Y
        if not return_only:
            if axes is None:
                fig  = plt.figure()
                axes = fig.add_subplot(111, projection='3d')
            axes.plot_trisurf(tri, Z, **kwargs)
            sz = np.max([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()])
            axes.set_xlim(X.mean()-sz*.6, X.mean()+sz*.6)
            axes.set_ylim(Y.mean()-sz*.6, Y.mean()+sz*.6)
            axes.set_zlim(Z.mean()-sz*.6, Z.mean()+sz*.6)
            plt.show()
        return tri, Z
    
    def intersect(self, photon_in, profiling=False, min_corrections=2, max_corrections=5):
        """Find intersection of incident light ray and the mirror surface.

        Arguments:
        photon_in - incident photons

        Returns:
        intersection - coordinate of the intersection, or (np.nan, np.nan, np.nan) if the intersection
                       does not exist.
        distance     - distance between the starting point and the intersection, or np.inf if the
                       intersection does not exist.
        """
        tic = time()
        # convert lab coordinates to mirror fixed coordinates.
        s,u = np.broadcast_arrays(
            quat.rotate(quat.conjugate(self.q), photon_in['position'].transpose()-self.p),
            quat.rotate(quat.conjugate(self.q), photon_in['direction'].transpose()))
        if profiling:
            toc = time()-tic
            print('coordinate transform: {:f} seconds'.format(toc))
            tic = time()
        # solve equations:
        # x = s[0] + u[0]*t,
        # y = s[1] + u[1]*t, and
        # z = s[2] + u[2]*t are linear equations, where t is distance between (x,y,z) and s.
        # f(x,y,z) = 0 is quadric surface equation.
        # substitute x, y and z with t in surface equation:
        # a t^2 + b t + c = 0.
        # if the equation holds for t>0, it means that the light ray from the starting point s
        # along direction u intersects with the surface, although not necessarily within the
        # region bounded by d_out and d_in.
        a = self.coef['x2']*u[0]**2.  + self.coef['y2']*u[1]**2.  + self.coef['z2']*u[2]**2. + \
            self.coef['xy']*u[0]*u[1] + self.coef['xz']*u[0]*u[2] + self.coef['yz']*u[1]*u[2]
        b = 2.*self.coef['x2']*s[0]*u[0] + \
            2.*self.coef['y2']*s[1]*u[1] + \
            2.*self.coef['z2']*s[2]*u[2] + \
            self.coef['xy']*(s[0]*u[1]+s[1]*u[0]) + \
            self.coef['xz']*(s[0]*u[2]+s[2]*u[0]) + \
            self.coef['yz']*(s[1]*u[2]+s[2]*u[1]) + \
            self.coef['x']*u[0] + \
            self.coef['y']*u[1] + \
            self.coef['z']*u[2]
        c = self.coef['x2']*s[0]**2.  + self.coef['y2']*s[1]**2.  + self.coef['z2']*s[2]**2. + \
            self.coef['xy']*s[0]*s[1] + self.coef['xz']*s[0]*s[2] + self.coef['yz']*s[1]*s[2] + \
            self.coef['x']*s[0] + self.coef['y']*s[1] + self.coef['z']*s[2] + self.coef['c']
        if profiling:
            toc = time()-tic
            print('setup 2nd-order equation: {:f} seconds'.format(toc))
            tic = time()
        a_is_0 = np.isclose(a, 0.)
        b_is_0 = np.isclose(b, 0.)
        t = np.zeros_like(s[0])
        # case 0: the line does not intersect the surface.
        t[  a_is_0  &   b_is_0 ] = np.inf
        minus_c_over_b = -c[a_is_0 & (~b_is_0)]/b[a_is_0 & (~b_is_0)]
        # case 1: equation f(t)=0 is linear.
        t[  a_is_0  & (~b_is_0)] = np.where(minus_c_over_b>=0., minus_c_over_b, np.inf)
        d = b**2. - 4.*a*c
        d_lt_0 = np.bool_(d<0.)
        # case 2: equation f(t)=0 is quadric but there is no real root.
        t[(~a_is_0) &   d_lt_0 ] = np.inf
        # case 3: equation f(t)=0 is quadric and there is two real roots.
        tp  = (-b[(~a_is_0)&(~d_lt_0)]+d[(~a_is_0)&(~d_lt_0)]**.5) / 2. / a[(~a_is_0)&(~d_lt_0)]
        tm  = (-b[(~a_is_0)&(~d_lt_0)]-d[(~a_is_0)&(~d_lt_0)]**.5) / 2. / a[(~a_is_0)&(~d_lt_0)]
        tpm = np.empty_like(tp)
        # case 3.1: both roots are negative, which means the light ray goes away from the intersections.
        tp_lt_0_and_tm_lt_0 = np.logical_and(np.bool_(tp<0.), np.bool_(tm<0.))
        tpm[tp_lt_0_and_tm_lt_0] = np.inf
        # case 3.2: one root is positive and the other one is negative, which means the starting point
        # lies between the two intersections.
        tp_lt_0_and_tm_gt_0 = np.logical_and(np.bool_(tp<0.), np.bool_(tm>=0.))
        tpm[tp_lt_0_and_tm_gt_0] = tm[tp_lt_0_and_tm_gt_0]
        tp_gt_0_and_tm_lt_0 = np.logical_and(np.bool_(tp>=0.), np.bool_(tm<0.))
        tpm[tp_gt_0_and_tm_lt_0] = tp[tp_gt_0_and_tm_lt_0]
        # case 3.3: both roots are positive, which means the light ray will intersect the surface twice.
        tp_gt_0_and_tm_gt_0 = np.logical_and(np.bool_(tp>=0.), np.bool_(tm>=0.))
        t1 = np.min([tp[tp_gt_0_and_tm_gt_0],tm[tp_gt_0_and_tm_gt_0]], axis=0)
        t2 = np.max([tp[tp_gt_0_and_tm_gt_0],tm[tp_gt_0_and_tm_gt_0]], axis=0)
        # case 3.3.1: the first intersection is out of boundaries, which means it misses the mirror.
        # case 3.3.1.1[out-of-boundaries case 1]: out of donut, i.e., r not in [r_in, r_out]
        # case 3.3.1.2[out-of-boundaries case 2]: wrong branch, i.e., z <= (g-f)/2.
        # case 3.3.2: the first intersection hits the mirror.
        n1 = s[:,(~a_is_0)&(~d_lt_0)][:,tp_gt_0_and_tm_gt_0] + \
            u[:,(~a_is_0)&(~d_lt_0)][:,tp_gt_0_and_tm_gt_0]*t1
        r1 = np.sum(n1[:2,:]**2., axis=0)**.5
        # merge case 3.3.1 and 3.3.2
        if self.g<0.:
            tpm[tp_gt_0_and_tm_gt_0] = np.where(
                (r1<=(self.d_out*.5))&\
                (r1>=(self.d_in*.5))&\
                (n1[2,:]>(self.g-self.f)/2.),
                t1, t2)
        else:
            tpm[tp_gt_0_and_tm_gt_0] = np.where(
                (r1<=(self.d_out*.5))&\
                (r1>=(self.d_in*.5)),
                t1, t2)
        # merge case 3.1, 3.2 and 3.3
        t[(~a_is_0)&(~d_lt_0)] = tpm[:]
        n = np.empty_like(s)
        t_is_inf = np.isinf(t)
        n[:, t_is_inf] = np.nan
        n[:,~t_is_inf] = s[:,~t_is_inf] + u[:,~t_is_inf]*t[~t_is_inf]
        if profiling:
            toc = time()-tic
            print('branching and mapping: {:f} seconds'.format(toc))
            tic = time()
        # rule out intersections on the other branch of hyperbolic surface
        if self.g<0:
            inside = np.bool_(n[2,~t_is_inf]>(self.g-self.f)/2.)
            t[  ~t_is_inf] = np.where(inside, t[  ~t_is_inf], np.inf)
            n[:,~t_is_inf] = np.where(inside, n[:,~t_is_inf], np.nan)
        # quadric error linear correction
        # quadric error is zero-mean bell-curve distributed error, typically around 1e-9
        k = 0
        t_is_inf = np.isinf(t)
        z = self.height(n[0,~t_is_inf], n[1,~t_is_inf])
        dz = (n[2,~t_is_inf]-z)
        while k<min_corrections or (k<max_corrections and not np.all(np.isclose(dz, 0.))):
            v = self.normal(np.double([n[0,~t_is_inf], n[1,~t_is_inf], z])) # normal vector
            cos_theta = v[2]/np.sum(v**2., axis=0)**.5
            cos_alpha = np.abs(np.sum(u[:,~t_is_inf]*v[:],axis=0)) / \
                quat.norm(u[:,~t_is_inf]) / quat.norm(v)
            dt = dz*cos_theta/cos_alpha
            t[  ~t_is_inf] = t[  ~t_is_inf]+dt
            n[:,~t_is_inf] = n[:,~t_is_inf]+dt*u[:,~t_is_inf]
            z = self.height(n[0,~t_is_inf], n[1,~t_is_inf])
            dz = (n[2,~t_is_inf]-z)
            k += 1
        # check out-of-boundary intersections.
        t_is_inf = np.isinf(t)
        rn = (n[0,~t_is_inf]**2. + n[1,~t_is_inf]**2.)**.5
        inside = np.bool_(rn<=(self.d_out*.5)) & \
            np.bool_(rn>=(self.d_in*.5))
        t[  ~t_is_inf] = np.where(inside, t[  ~t_is_inf], np.inf)
        n[:,~t_is_inf] = np.where(inside, n[:,~t_is_inf], np.nan)
        return n, t

    def height(self, x, y):
        """Get z-coordinate (height) of surface at (x,y) in fixed csys.
        """
        z = np.empty_like(x)
        a = self.coef['z2']
        b = self.coef['xz']*x + self.coef['yz']*y + self.coef['z']
        c = self.coef['x2']*x**2. + self.coef['y2']*y**2. + \
            self.coef['xy']*x*y + \
            self.coef['x']*x + self.coef['y']*y + self.coef['c']
        b_is_0 = np.isclose(b, 0.)
        if np.isclose(a, 0.):
            z[ b_is_0] = np.nan
            z[~b_is_0] = -c[~b_is_0] / b[~b_is_0]
            return z
        else:
            d = -(b**2. - 4.*a*c)**.5
        z = (-b+d)/2./a
        return z
    
    def normal(self, r):
        """Normal vector at position r of the surface.
        A normal vector is a unit vector from given position at the surface towards the focus,
        i.e., along +z direction and perpendicular to the tangential plane of the surface.

        Argument:
        r   - coordinate in mirror fixed csys.

        Return:
        n   - unit normal vector in fixed csys.
        """
        # r = quat.rotate(quat.conjugate(self.q), r-self.p)
        # d={
        #     'x2':self.coef['x2'],
        #     'y2':self.coef['y2'],
        #     'z2':self.coef['z2'],
        #     'xy':self.coef['xy'],
        #     'yz':self.coef['yz'],
        #     'xz':self.coef['xz'],
        #     'x' :self.coef['x' ],
        #     'y' :self.coef['y' ],
        #     'z' :self.coef['z' ],
        #     'rx':r[0],
        #     'ry':r[1],
        #     'rz':r[2]
        # }
        # n = np.double([
        #     ne.evaluate('x2*2.*rx+xy*ry+xz*rz+x', local_dict=d),
        #     ne.evaluate('y2*2.*ry+xy*rx+yz*rz+y', local_dict=d),
        #     ne.evaluate('z2*2.*rz+yz*ry+xz*rx+z', local_dict=d)])
        n = np.double([
            self.coef['x2']*2.*r[0]+self.coef['xy']*r[1]+self.coef['xz']*r[2]+self.coef['x'],
            self.coef['y2']*2.*r[1]+self.coef['xy']*r[0]+self.coef['yz']*r[2]+self.coef['y'],
            self.coef['z2']*2.*r[2]+self.coef['yz']*r[1]+self.coef['xz']*r[0]+self.coef['z']])
        if np.isinf(self.f):
            n = np.where(n[2]>0., n, -n)
        else:
            n = np.where(np.sum(n*r, axis=0)<0., n, -n)
        return n
    
    def encounter(self, photon_in, intersection):
        """Determine outcomes when a super photon encounters with the mirror.
        Validation of intersections should be guaranteed in advance.

        Outcomes:
        reflected photon   - same weight as incident photon
                             position at intersection
                             mirrored direction
                             phase determined from optical path lenght
        absorbed photon    - same weight as incident photon
                             position at intersection
                             zeroed direction
                             phase determined from optical path length
        transmitted photon - same weight as incident photon
                             position at intersection
                             same direction as incident photon
                             phase determined from optical path length

        Arguments:
        photon_in    - incident super photons, with position and direction vectors in lab csys.
        intersection - corresponding intersections.

        Return:
        photon_out   - outcome super photons, with position and direction vectors in lab csys.
        """
        photon_out = np.empty_like(photon_in)
        n = quat.rotate(self.q, self.normal(intersection)) # normal vector in lab csys
        phi_n, theta_n, _ = quat.xyz2ptr(n[0], n[1], n[2]) # longtitude and latitude of normal vector in lab csys
        # attitude Euler's angles
        phi_a   = np.where(theta_n>0., phi_n-np.pi, phi_n)
        theta_a = np.where(theta_n>0., np.pi/2.-theta_n, np.pi/2.+theta_n)
        q = quat.from_angles(phi_a, theta_a)
        photon_out['weight'][:]     = photon_in['weight']
        photon_out['wavelength'][:] = photon_in['wavelength']
        photon_out['position'][:]   = np.transpose(quat.rotate(self.q, intersection)+self.p)
        p = np.transpose(photon_out['position'] - photon_in['position'])
        u = quat.rotate(quat.conjugate(q), photon_in['direction'].transpose())
        d = np.reshape(p[0]*n[0] + p[1]*n[1] + p[2]*n[2], (-1, 1))
        top_mask = np.double([np.abs(self.boundary[0]), np.abs(self.boundary[0]), -self.boundary[0]])
        bot_mask = np.double([np.abs(self.boundary[1]), np.abs(self.boundary[1]), -self.boundary[1]])
        v = u.transpose()*np.where(d<0., top_mask, bot_mask)
        photon_out['direction'][:] = quat.rotate(q, v.transpose()).transpose()
        photon_out['phase'][:]     = photon_in['phase'] + \
            2.*np.pi*np.sum(p.transpose()**2.,axis=1)**.5/photon_in['wavelength']
        photon_out['position'][:] += dm*photon_out['direction']
        photon_out['phase'][:]     = np.mod(photon_out['phase'][:]+2.*np.pi*dm, 2.*np.pi)
        return photon_out

class OpticalSystem(object):
    """Optical system.
    """
    def __init__(self, p=[0., 0., 0.], q=[1., 0., 0., 0.], name=''):
        """Create empty optical system.

        Arguments:
        p    - position in lab coordinate system.
        q    - attitude quaternion in lab coordinate system.
        name - human readable name assigned to the system.

        Example:
        r' = qrq' + p,
        where r is vector in system fixed csys and r' is vector in lab csys.
        """
        self.p = np.double(p).reshape((3,1))
        self.q = np.double(q).reshape((4,1))
        self.parts = []
        self.name = name
        self.raytracing_multiplier = 2 # max_raytracing_steps = raytracing_multiplier * num_parts

    def get_entrance(self):
        """Get entrance apertures.
        """
        entrance = []
        for p in self.parts:
            if p.is_entrance:
                entrance.append(p)
        return entrance

    def get_primaries(self):
        """Get primary mirrors.
        """
        primaries = []
        for p in self.parts:
            if p.is_primary:
                primaries.append(p)
        return primaries

    def get_virtual_parts(self):
        """Get virtual parts.
        """
        virtuals = []
        for p in self.parts:
            if p.is_virtual:
                virtuals.append(p)
        return virtuals

    def get_detectors(self):
        """Get detectors.
        """
        detectors = []
        for p in self.parts:
            if p.is_detector:
                detectors.append(p)
        return detectors

    def get_parts_by_name(self, name):
        """Get parts by name.
        """
        parts = []
        for p in self.parts:
            if p.name == name:
                parts.append(p)
        return parts
    
    def add_part(self, part):
        """Append SymmetricQuadricMirror object to the current system.
        """
        self.parts.append(part)

    def join(self, guest_system):
        """Join all parts of an existing guest OpticalSystem object to the
        current object (host).
        """
        for part in guest_system.parts:
            if part not in self.parts:
                self.add_part(part)
    
    def move(self, dp):
        """Move system in lab csys by displacement vector dp.

        new position: p' = p + dp
        """
        for m in self.parts:
            m.p = m.p + np.double(dp).reshape((3,1))
        self.p = self.p + np.double(dp).reshape((3,1))
        return self.p
    
    def rotate(self, dq):
        """Rotate system around its fixed csys origin by dq.

        new attitude: q' = dq q dq'
        """
        for m in self.parts:
            m.p = quat.rotate(np.double(dq).reshape((4,1)), m.p)
            m.q = quat.multiply(np.double(dq).reshape((4,1)), m.q)
        self.q = quat.multiply(np.double(dq).reshape((4,1)), self.q)
        return self.q
    
    def set_p(self, p):
        """Move to new position p.
        """
        dp = np.double(p).reshape((3,1))-self.p
        self.move(dp)
        
    def set_q(self, q):
        """Rotate to new attitude q.
        """
        dq = quat.multiply(np.double(q).reshape((4,1)), quat.conjugate(self.q))
        self.rotate(dq)

    def draw(
            self,
            nside=32,
            axes=None,
            figure_size=None,
            visualizer='matplotlib',
            return_only=False,
            draw_virtual=True,
            highlight_primary=True,
            highlight_list=[],
            raytrace=None,
            view_angles=(0., -90.),
            virtual_opts={},
            highlight_opts={},
            surface_opts={},
            lightray_opts={},
            axes_fontsize=12,
            output=None
    ):
        """Draw triangular surface of all mirrors.

        Arguments:
        nside              - nside of healpix sampling points for each mirror.
        axes               - matplotlib.axes3D object.
        figure_size        - figure size, by (width, height), in pixels.
        visualizer         - 3D visualizer, mayavi (default) or matplotlib (fallback).
        return_only        - instead of plotting the surface, return the triangles only (boolean switch).
        draw_virtual       - draw virtual mirrors (boolean switch).
        highlight_primary  - highlight primary mirrors (boolean switch).
        highlight_list     - list of indices of highlight mirrors.
        raytrace           - plot ray trace (optional).
        view_angles        - view angles, by (elevation, azimuth), in degrees.
                             elevation is the angle between line of sight and xy plane.
                             azimuth is the angle between line of sight and xz plane.
        virtual_opts       - plot_trisurf (or mayavi.mlab.triangle_surf) keyword options for virtual surfaces, e.g., entrance aperture.
        highlight_opts     - plot_trisurf (or mayavi.mlab.triangle_surf) keyword options for highlighted surfaces, e.g., primary mirror.
        surface_opts       - plot_trisurf (or mayavi.mlab.triangle_surf) keyword options for ordinary surfaces.
        lightray_opts      - plot (or mayavi.mlab.plot3d) keyword options for light rays.
        axes_fontsize      - font size for axes, such as labels.
        output             - figure output. Default: None, i.e., immediately shows on screen.
        """
        trigs = []
        Zs    = []
        names = []
        xmin  = 0.
        xmax  = 0.
        ymin  = 0.
        ymax  = 0.
        zmin  = 0.
        zmax  = 0.
        for m in self.parts:
            trig, Z = m.draw(nside=nside, return_only=True)
            trigs.append(trig)
            Zs.append(Z)
            xmin = min(xmin, trig.x.min())
            xmax = max(xmax, trig.x.max())
            ymin = min(ymin, trig.y.min())
            ymax = max(ymax, trig.y.max())
            zmin = min(zmin, Z.min())
            zmax = max(zmax, Z.max())
        sz = np.max([xmax-xmin, ymax-ymin, zmax-zmin])
        xc = .5*(xmin+xmax)
        yc = .5*(ymin+ymax)
        zc = .5*(zmin+zmax)
        extent = {'x':(xmin, xmax),
                  'y':(ymin, ymax),
                  'z':(zmin, zmax)}
        colors = {
            'virtual'  :(.776, .886, .999),
            'highlight':(.745, .062, .075),
            'lightray' :(.000, .500, .000),
            'surface'  :(.999, .859, .000)
        }
        if figure_size is None:
            figure_size = (1200, 1200)
        if not return_only:
            if visualizer.lower().startswith('maya'):
                if len(virtual_opts)==0:
                    virtual_opts={
                        'opacity': 0.3,
                        'color': colors['virtual'],
                        'resolution': 8
                    }
                if len(highlight_opts)==0:
                    highlight_opts={
                        'opacity': 1.0,
                        'color': colors['highlight'],
                        'resolution': 8
                    }
                if len(lightray_opts)==0:
                    lightray_opts={
                        'opacity':    0.05,
                        'line_width': 0.01,
                        'color':     colors['lightray'],
                        'representation':'surface'
                    }
                if len(surface_opts)==0:
                    surface_opts={
                        'opacity': 1.0,
                        'color': colors['surface'],
                        'resolution': 8
                    }
                if output is None:
                    mlab.options.offscreen = False
                else:
                    mlab.options.offscreen = True
                fig = mlab.figure(bgcolor=(1., 1., 1.), size=(1024, 1024))
                mlab.view(azimuth=view_angles[1], elevation=90.+view_angles[0])
                for i in range(len(self.parts)):
                    if self.parts[i].is_virtual:
                        if draw_virtual:
                            mobj = mlab.triangular_mesh(trigs[i].x, trigs[i].y, Zs[i], trigs[i].triangles[~trigs[i].mask], **virtual_opts)
                            mobj.actor.property.lighting = False
                        continue
                    if (i in highlight_list) or (highlight_primary and self.parts[i].is_primary):
                        mobj = mlab.triangular_mesh(trigs[i].x, trigs[i].y, Zs[i], trigs[i].triangles[~trigs[i].mask], **highlight_opts)
                        mobj.actor.property.lighting = False
                        continue
                    mobj = mlab.triangular_mesh(trigs[i].x, trigs[i].y, Zs[i], trigs[i].triangles[~trigs[i].mask], **surface_opts)
                if raytrace is not None:
                    nodes, npts = raytrace.shape
                    for i in range(npts):
                        mobj = mlab.plot3d(
                            raytrace['position'][:,i,0],
                            raytrace['position'][:,i,1],
                            raytrace['position'][:,i,2],
                            **lightray_opts
                        )
                        mobj.actor.property.lighting = False
                if output is None:
                    mlab.show()
                else:
                    mlab.savefig(output, magnification=2)
            elif visualizer.lower().startswith('mat'):
                if len(virtual_opts)==0:
                    virtual_opts={
                        'alpha': 0.5,
                        'color': colors['virtual'],
                        'linestyle': 'None'
                    }
                if len(highlight_opts)==0:
                    highlight_opts={
                        'alpha': 1.0,
                        'color': colors['highlight'],
                        'linestyle': 'None'
                    }
                if len(lightray_opts)==0:
                    lightray_opts={
                        'alpha':     0.5,
                        'linewidth': 0.05,
                        'color':     colors['lightray']
                    }
                if len(surface_opts)==0:
                    surface_opts={
                        'alpha': 1.0,
                        'color': colors['surface'],
                        'linestyle': 'None'
                    }
                if axes is None:
                    fig = plt.figure(figsize=(figure_size[0]/rcParams['figure.dpi'], figure_size[1]/rcParams['figure.dpi']))
                    axes = fig.add_subplot(111, projection='3d')
                axes.view_init(*view_angles)
                for i in range(len(self.parts)):
                    if self.parts[i].is_virtual:
                        if draw_virtual:
                            mobj = axes.plot_trisurf(trigs[i], Zs[i], **virtual_opts)
                        continue
                    if (i in highlight_list) or (highlight_primary and self.parts[i].is_primary):
                        mobj = axes.plot_trisurf(trigs[i], Zs[i], **highlight_opts)
                        continue
                    mobj = axes.plot_trisurf(trigs[i], Zs[i], **surface_opts)
                if raytrace is not None:
                    nodes, npts = raytrace.shape
                    for i in range(npts):
                        mobj = axes.plot(
                            raytrace['position'][:,i,0],
                            raytrace['position'][:,i,1],
                            raytrace['position'][:,i,2],
                            **lightray_opts
                        )
                axes.set_xlim(xc-.5*sz, xc+.5*sz)
                axes.set_ylim(yc-.5*sz, yc+.5*sz)
                axes.set_zlim(zc-.5*sz, zc+.5*sz)
                axes.set_xlabel('x, in meters', fontsize=axes_fontsize)
                axes.set_ylabel('y, in meters', fontsize=axes_fontsize)
                axes.set_zlabel('z, in meters', fontsize=axes_fontsize)
                plt.axis('off')
                plt.tight_layout()
                if output is None:
                    plt.show()
                else:
                    plt.savefig(output, dpi=rcParams['figure.dpi']*2)
                    plt.close()
        return trigs, Zs, extent
    
    def intersect(self, photon_in, parts=None):
        """Find intersections of incident light rays and the mirrors.

        Arguments:
        photon_in - incident super photons.
        parts     - list of all parts the light rays may encouter.

        Returns:
        n - intersections, in lab csys.
        t - optical path lengths between starting points and intersections.
        k - indices of mirror encountered.
        """
        if parts is None:
            parts = self.parts
        n = np.empty((3, photon_in.size), dtype='double')
        t = np.empty((photon_in.size,  ), dtype='double')
        k = np.empty((photon_in.size,  ), dtype=mstype  )
        n[:] = np.nan
        t[:] = np.inf
        k[:] = -1
        for i in range(len(parts)):
            ni, ti = parts[i].intersect(photon_in)
            m = np.bool_(ti<t)
            n[:] = np.where(m, ni, n)
            t[:] = np.where(m, ti, t)
            k[:] = np.where(m, self.parts.index(parts[i]), k)
        return n, t, k

    def trace(self, photon_in, profiling=False):
        """Trace photons in optical system.

        Arguments:
        photon_in - input photons
        profiling - print simple profiling messages or not

        Returns:
        photon_trace - photon trace at each step
        mirror_trace - optical parts encountered at each step

        Both photon_trace and mirror_trace are numpy array
        in (nvtx, npts), where nvtx is the number of vertices
        and npts is the number of tracked super photons.
        Each vertex is a snapshot of the photon along its trace.
        The first vertex is a replicate of the input photon.
        The second vertex is the outcome after a photon encounters
        the first optical parts along its trace.
        Say there are N segments between adjacent optical parts
        along a trace, as a result there are N+2 vertices in total.
        """
        photon_trace = []
        mirror_trace = []
        msbuf = np.empty((photon_in.size,), dtype=mstype)
        spbuf = np.copy(photon_in)
        msbuf[:] = -1
        photon_trace.append(np.copy(spbuf))
        mirror_trace.append(np.copy(msbuf))
        i = 0
        max_raytracing_steps = len(self.parts) * self.raytracing_multiplier
        while i < max_raytracing_steps:
            tic = time()
            n,t,k = self.intersect(photon_trace[i])
            if profiling:
                toc = time()-tic
                print('computing intersection: {:f} seconds'.format(toc))
            msbuf[:] = k
            if np.allclose(msbuf, -1):
                break
            mirror_trace.append(np.copy(msbuf))
            m = np.isinf(t)
            spbuf[m] = photon_trace[i][m]
            hits = np.bincount(k[~m], minlength=len(self.parts))
            for l in range(len(self.parts)):
                tic = time()
                if hits[l]>0:
                    m = np.bool_(k==l)
                    spbuf[m] = self.parts[l].encounter(photon_trace[i][m], n[:, m])
                if profiling:
                    toc = time()-tic
                    print('computing outcomes: {:f} seconds'.format(toc))
            photon_trace.append(np.copy(spbuf))
            i += 1
        return np.concatenate(photon_trace).reshape((-1,photon_in.size)), \
            np.concatenate(mirror_trace).reshape((-1,photon_in.size))

    def trace_network(
            self,
            photon_in,
            network,
            reverse=False,
            starts=None,
            stops=None,
            profiling=False):
        """Trace photons in optical system via previously built optical path network.

        Arguments:
        photon_in - Input photons.
        network   - Previously built optical path network (OPN).
        reverse   - Given OPN is a directed acyclic graph, trace photons along the
                    direction of each edge or reversely.
        starts    - List of starting nodes.
        stops     - List of stopping nodes.
        profiling - Print simple profiling messages or not.

        Returns:
        photon_trace - snapshots of photons at each step
        mirror_trace - optical parts encountered at each step
        """
        if starts is None:
            starts = network.entrance_nodes()
        if stops is None:
            stops = network.exit_nodes()
        paths = []
        for start in starts:
            for stop in stops:
                if reverse:
                    paths += [p for p in nx.all_simple_paths(network, stop, start)]
                else:
                    paths += [p for p in nx.all_simple_paths(network, start, stop)]
        max_length = 0
        for p in paths:
            max_length = max(max_length, len(p))
        nvtx = max_length + 1
        npts = photon_in.size
        photon_trace      = np.empty((nvtx, npts), dtype=sptype)
        mirror_trace      = np.empty((nvtx, npts), dtype=mstype)
        photon_trace[0,:] = photon_in[:]
        # layer 0, free space
        mirror_trace[:,:] = -1
        # layer 1, entrance
        photon_trace[1,:] = photon_trace[0,:]
        to_parts = starts
        n,t,k = self.intersect(photon_trace[0,:], parts=to_parts)
        for p in to_parts:
            ip = self.parts.index(p)
            mp = np.bool_(k==ip)
            photon_trace[1,mp] = p.encounter(photon_trace[0,mp], n[:,mp])
            mirror_trace[1,mp] = ip
        # layer 2 to N
        from_parts = to_parts
        for l in range(2, nvtx):
            photon_trace[l,:] = photon_trace[l-1,:]
            to_parts = []
            for fp in from_parts:
                if fp in stops:
                    continue
                if reverse:
                    sub_to_parts = [p for p in network.predecessors(fp)]
                else:
                    sub_to_parts = [p for p in network.successors(fp)]
                to_parts += sub_to_parts
                ifp = self.parts.index(fp)
                mfp = np.bool_(mirror_trace[l-1,:]==ifp)
                n,t,k = self.intersect(photon_trace[l-1,mfp], parts=sub_to_parts)
                ptbuf = np.copy(photon_trace[l,mfp])
                mtbuf = np.copy(mirror_trace[l,mfp])
                for tp in sub_to_parts:
                    itp = self.parts.index(tp)
                    mtp = np.bool_(k==itp)
                    ptbuf[mtp] = tp.encounter(photon_trace[l-1,mfp][mtp], n[:,mtp])
                    mtbuf[mtp] = itp
                photon_trace[l,mfp] = ptbuf
                mirror_trace[l,mfp] = mtbuf
            from_parts = to_parts
        return photon_trace, mirror_trace

class OpticalPathNetwork(nx.classes.digraph.DiGraph):
    """An optical path network is a directed acyclic graph (DAG) that represents
    optical paths between components of a given optical system and a light source.

    Optical system high-level modelling and analysis tools:
    1. draw()           - optical path visualization
    2. entrance_nodes() - get entrance nodes, such as primary mirror, stray light
                          baffle and so on, by non-sequential raytracing
    3. exit_nodes()     - get exit nodes, such as focal plane, stray light absorber
                          and so on, by non-sequential raytracing
    
    """
    def __init__(
            self,
            optical_system,
            light_source=None,
            photon_trace=None,
            mirror_trace=None,
            title='',
            description=''):
        """Build optical path network that the light travels from light source in
        the system. The underlying object for the optical path network is directed
        graph implemented with NetworkX.


        Arguments:
        optical_system - Underlying physical entity of the optical path network.
        light_source   - Reference light source, default: point light source at (0,0,inf)
                         that emits planar wave along -Z axis.
        photon_trace   - Super photon trace.
                         each trace is tracked by a series of snapshots of the super photon
                         as it encounters a PART of the optical system.
        mirror_trace   - Mirror trace corresponds to the photon trace.
        title          - Title of the network.
        description    - Additional comments on the network.
        """
        if light_source is None:
            light_source = LightSource((0., 0., np.inf))
        if (photon_trace is None) or (mirror_trace is None):
            # trace light rays from given light source
            p, q = light_source(optical_system.get_entrance(), 100, 1, sampling='random')
            photon_trace, mirror_trace = optical_system.trace(q)
        super(OpticalPathNetwork, self).__init__(
            light_source=light_source,
            optical_system=optical_system,
            title=title,
            description=description)
        nvtx, npts = mirror_trace.shape
        for p in optical_system.parts:
            if not p.is_virtual:
                self.add_node(p)
        for i in range(1, nvtx-1):
            # encode (from_part_index, to_part_index)
            m = (np.isclose(mirror_trace[i], -1) | np.isclose(mirror_trace[i+1], -1))
            edge_codes = np.unique(mirror_trace[i,~m]*len(optical_system.parts)+mirror_trace[i+1,~m])
            for ec in edge_codes:
                to_idx = np.mod(ec, len(optical_system.parts))
                from_idx = (ec-to_idx)//len(optical_system.parts)
                self.add_edge(optical_system.parts[from_idx], optical_system.parts[to_idx])

    def draw(self, pos=None, output=None, **kwargs):
        """Draw optical path network.

        pos      - position of nodes
        output   - output of figure (default: None, to print on the screen)
        **kwargs - keyword options to pass to networkx.draw()
        """
        if pos is None:
            pos = nx.spring_layout(self, pos=nx.kamada_kawai_layout(self), iterations=2)
        if 'node_color' not in kwargs:
            kwargs['node_color'] = 'navy'
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 7
        if 'font_color' not in kwargs:
            kwargs['font_color'] = 'lightgray'
        nx.draw_networkx(self, pos=pos, with_labels=True, labels={node:node.name for node in self.nodes}, **kwargs)
        if output is None:
            plt.show()
        else:
            plt.savefig(output)
            plt.close()

    def entrance_nodes(self):
        """Get entrance nodes if there is any.
        An entrance node get photons directly and only from the light source.
        """
        nodes = []
        for node in self.nodes:
            if self.in_degree(node)==0:
                nodes.append(node)
        return nodes
    
    def exit_nodes(self):
        """Get exit nodes if there is any.
        An exit node absorbs or scatters all photons so that no other node get
        any photon from it.
        """
        nodes = []
        for node in self.nodes:
            if self.out_degree(node)==0:
                nodes.append(node)
        return nodes

    def aperture_stop(self, on_axis_source=None, rays=1000):
        """Find all aperture stop(s) of the underlying optical system.

        Arguments:
        on_axis_source - The on-axis light source. The aperture stop limits the
                         marginal rays from an on-axis light source by definition.
                         Default: the light source used to construct the optical
                         path network, i.e., self.graph['light_source'].
        rays           - Number of rays (Default: 1,000).
        """
        if on_axis_source is None:
            on_axis_source = self.graph['light_source']
        _,q = on_axis_source(self.entrance_nodes(), rays, 1., sampling='crosshair')
        pt,mt = self.graph['optical_system'].trace_network(q, self)
        stops = [] # list of aperture stops
        from_parts = self.entrance_nodes()
        nvtx, npts = pt.shape
        for l in range(1,nvtx-1):
            to_parts = []
            for fp in from_parts:
                sub_to_parts = [node for node in self.successors(fp)]
                to_parts += sub_to_parts
                ifp = self.graph['optical_system'].parts.index(fp)
                mfp = np.bool_(mt[l,:]==ifp)
                if np.any(mt[l+1,mfp]==-1):
                    stops += sub_to_parts
            from_parts = to_parts
        return stops
 
    def field_stop(self):
        pass
    
    def entrance_pupil(
            self,
            aperture_stops=None,
            on_axis_source=None,
            batch_rays=10000,
            perturbation=3.,
            min_samplings=1000,
            max_batches=10,
            verbose=False):
        """Find entrance pupil of the underlying optical system.

        Arguments:
        aperture_stops  - List of aperture stops.
        on_axis_source  - The on-axis light source used to determine
                          the aperture stop(s). By definition the
                          marginal rays from an on-axis source is limited
                          by aperture stop(s).
        batch_rays      - Number of rays per batch.
        perturbation    - Perturbation size in term of standard deviation,
                          in degrees.
                          Light rays hit aperture stops from on-axis source
                          is perturbed before traced backwards to entrance
                          of the optical system in order to find intersections
                          between rays from the same locations on the aperture
                          stops.
        min_samplings   - Minimum number of sampling points required
                          on the entrance pupil.
        max_batches     - Maximum number of batches.
                          Process will be terminated when it reaches this limit
                          and RuntimeError is raised.
        verbose         - Print verbose messages for diagnosis.
        """
        if on_axis_source is None:
            on_axis_source = self.graph['light_source']
        if aperture_stops is None:
            aperture_stops = self.aperture_stop(on_axis_source=on_axis_source)
        entrance = self.entrance_nodes()
        num_samplings = 0
        samplings = []
        batch = 0
        optics = self.graph['optical_system']
        while num_samplings < min_samplings:
            if batch < max_batches:
                batch += 1
            else:
                raise RuntimeError("Limit max_batches reached. Try increase max_batches or decrease min_samplings.")
            _,q = on_axis_source(
                entrance,
                batch_rays, 1.,
                sampling='random') # sampled super photons from on-axis source
            pt0, mt0 = optics.trace_network(
                q, self, stops=aperture_stops) # forward raytracing
            m0 = np.zeros(mt0[0,:].shape, dtype='bool') # boolean mask
            m0[:] = False
            for s in aperture_stops:
                m0[:] = (m0[:] | (mt0[-1,:]==optics.parts.index(s)))
            p = pt0[-1,m0]
            if verbose:
                print("Batch {:d}: {:d} rays traced on aperture stop.".format(batch, p.size))
            p['direction'][:] = gimbal(
                -p['direction'][:],
                np.pi*np.random.rand(*p.shape),
                np.random.normal(
                    np.zeros(p.shape),
                    np.deg2rad(perturbation)))
            if verbose:
                print("Batch {:d}: {:d} rays perturbed.".format(batch, p.size))
            pt1, mt1 = optics.trace_network(
                p, self, reverse=True,
                starts=aperture_stops,
                stops=entrance)
            m1 = np.zeros(mt1[0,:].shape, dtype='bool')
            m1[:] = False
            for s in entrance:
                m1[:] = (m1[:] | (mt1[-1,:]==optics.parts.index(s)))
            p = pt1[-1,m1]
            if verbose:
                print("Batch {:d}: {:d} backwards rays traced on entrance.".format(batch, p.size))
            n, s = lins.two_lines_intersection(
                pt0[0,m0][m1]['position'],
                pt0[0,m0][m1]['direction'],
                p['position'],
                p['direction'])
            m2 = ~np.isnan(s[:])
            if verbose:
                print("Batch {:d}: {:d} intersections solved, S-stats: {:.2E} (min), {:.2E} (max), {:.2E} (avg).".format(batch, s[m2].size, np.min(s[m2]), np.max(s[m2]), np.mean(s[m2])))
            samplings.append(n[m2,:])
            num_samplings += s[m2].size
        return np.concatenate(samplings)
            
    def exit_pupil(self):
        pass

    def entrance_window(self):
        pass

    def exit_window(self):
        pass
            
class Detector(SymmetricQuadricMirror):
    """Pixel-array photon detector model.
    Geometrically a detector is a innerscribed square of an aperture of a given
    optical system. The aperture where the detector is installed is called its
    circumcircle aperture. The effective area of the detector is its incirle
    aperture.
    """
    def __init__(
            self,
            a,
            N,
            p=[0., 0., 0.],
            q=[1., 0., 0., 0.],
            name='unnamed detector'
    ):
        """Create a pixel-array photon detector object.
        a    - size of the detector, in meter
        N    - number of pixels along each axis of the detector
        p    - position of the center of the detector, in lab csys
        q    - attitude quaternion of the detector, in lab csys
        name - user defined name
        """
        super(Detector, self).__init__(
            0.,
            a*1.414213562373095,
            f=np.inf,
            g=np.inf,
            b=(0, 0),
            p=np.double(p).reshape((3,1)),
            q=np.double(q).reshape((4,1)),
            name=name,
            is_virtual=False,
            is_entrance=False,
            is_primary=False,
            is_detector=True)
        self.optics = None
        self.npxs   = N
        self.a      = a
        xi = ((np.arange(N)+.5)/N - .5)*a
        yi = ((np.arange(N)+.5)/N - .5)*a
        self.x, self.y = np.meshgrid(xi, yi)
        self.photon_buffer = np.empty((0,), dtype=sptype)

    def encounter(self, photon_in, intersection):
        """Overriding SymmetricQuadricMirror.encounter()
        """
        photon_out = super(Detector, self).encounter(photon_in, intersection)
        photon_det = np.copy(photon_out)
        photon_det['position'][:] = intersection.transpose()
        self.photon_buffer = np.concatenate((self.photon_buffer, photon_det))
        return photon_out
    
    def readout(self, clear_buffer=True):
        """Register buffered photon events to pixel grid, get binned counts
        and clear photon buffer.
        """
        xid = np.int64((self.photon_buffer['position'][:,0]+self.a/2.)/self.a*self.npxs)
        yid = np.int64((self.photon_buffer['position'][:,1]+self.a/2.)/self.a*self.npxs)
        pid = yid*self.npxs+xid
        re  = (self.photon_buffer['weight']**.5)*np.cos(self.photon_buffer['phase'])
        im  = (self.photon_buffer['weight']**.5)*np.sin(self.photon_buffer['phase'])
        remap = np.bincount(pid.ravel(), weights=re, minlength=self.npxs*self.npxs)
        immap = np.bincount(pid.ravel(), weights=im, minlength=self.npxs*self.npxs)
        amp = np.reshape((remap**2.+immap**2.)**.5, (self.npxs, self.npxs))
        pha = np.ma.masked_array(np.reshape(np.arctan2(immap, remap), (self.npxs, self.npxs)), mask=~(amp>0.))
        if clear_buffer:
            self.photon_buffer = np.empty((0,), dtype=sptype)
        return amp, pha
    
class CassegrainReflector(OpticalSystem):
    def __init__(self):
        pass
    
class PhotonCollector(OpticalSystem):
    def __init__(self, d=2., f=4., r=10., fov=np.deg2rad(5./60.)):
        super(PhotonCollector, self).__init__()
        primary_f      = f
        primary_d_out  = d
        secondary_f    = f/r
        beam_d         = d/r
        primary_d_in   = beam_d + 2.*np.tan(fov*.5)*f*r
        secondary_d    = primary_d_in + 2.*np.tan(fov*.5)*f
        entrance_d_out = primary_d_out + 2.*np.tan(fov*.5)*f
        entrance_d_in  = secondary_d
        a0 = SymmetricQuadricMirror(entrance_d_in, entrance_d_out, f=np.inf,      b=(-1,-1), g=np.inf, is_entrance=True, is_virtual=True)
        m0 = SymmetricQuadricMirror(primary_d_in,  primary_d_out,  f=primary_f,   b=( 1, 0), g=np.inf, is_primary=True)
        m1 = SymmetricQuadricMirror(0,             secondary_d,    f=secondary_f, b=( 0, 1), g=np.inf)
        m2 = SymmetricQuadricMirror(0,             secondary_d*2., f=np.inf,      b=( 1, 0), g=np.inf, p=[0., 0., -primary_f-2*secondary_d], q=quat.from_angles(0., -np.pi/4.))
        self.add_part(a0)
        self.add_part(m0)
        self.add_part(m1)
        self.add_part(m2)

class SIM(OpticalSystem):
    """Simplified Michelson stellar interferometer model.
    
    1. Overall model
    
          M10                                      M11
     |    --   |                              |    --    |
     |   /| \  |                              |   / |\   |
     |  / ||---+------------- B --------------+----|| \  |
     | /  |   \|          |-- B' --|          | /   |  \ |
     ---- |  ---- M00            M7       M01 ----  |  ----
          |            M50   ----   M51             |
          |       M40 /---\  |  |  /---\M41         |
          |           |   | /|  |\ |   |            |
          |           |   |/ |  | \|   |            |
          |           |   ---\  /---   |            |
          \-----------/   M6  ||       \------------/
         M20         M30      \/        M31         M21
                              -- D0
    
    M00 (M01) - Concave parabolic mirror, primary mirror of left (right) collector.
    M10 (M11) - Convex parabolic mirror, secondary mirror of left (right) collector.
    M20 (M21) - Plane mirror, periscope mirror of left (right) collector.
    M30, M31  - Plane mirrors, periscope mirrors of beam combiner.
    M40, M50  - Plane mirrors, delay line controller of the left arm.
    M41, M51  - Plane mirrors, delay line controller of the right arm.
    M6        - Concave parabolic mirror, primary mirror of beam combiner.
    M7        - Convex hyperbolic mirror, secondary mirror of beam combiner.
    D0        - Pixel detector array.
    
    2. Structual models
    2.1 Left collector
    The left collector contains M00, M10 and M20.
    M20 has tip and tilt actuators.
    2.2 Right collector
    The right collector contains M01, M11 and M21.
    M21 also has tip and tilt actuators.
    2.3 Beam combiner
    The beam combiner contains M30, M31, M40, M41, M50, M51, M6, M7 and D0.
    M30 and M31 both have tip and tilt actuators.
    M40 and M50 are coupled and have a shared piston actuator.
    M41 and M51 are coupled and have a shared piston actuator.
    2.4 Degrees of freedom
    Each structure has 3 translational as well as 3 rotational degrees of freedom.
    
    3. Optical models
    3.1 Photon collecting telescopes
    M00 (M01) and M10 (M11) constitute the left (right) collecting telescope.
    3.2 Periscopes
    M20 (M21) and M30 (M31) constitute the left (right) periscope.
    3.3 Delay line controllers
    M40 (M41) and M50 (M51) constitute the left (right) delay line controller.
    3.4 Beam combining telescope
    M6 and M7 constitute the beam combining telescope.
    
    4. Coordinate systems
    4.1 Coordinate system of beam combiner as well as the whole interferometer
    origin: focus of primary mirror.
    z-axis: along principal optical axis of beam combining telescope, from D0 to M5.
    y-axis: along the interferometer baseline, from M30 to M31.
    4.2 Coordinate system of collector
    origin: focus of primary mirror.
    z-axis: along principal optical axis of beam compressor, from M10 (M11) to M20 (M21).
    x-axis: along beam reflected by M20 (M21).
    """
    def __init__(
            self,
            collector_d=2.,
            collector_r=5.,
            collector_f=5.,
            combiner_b=2.,
            combiner_r=5.,
            combiner_f=5.,
            optics_fov=np.deg2rad(10./60.),
            detector_a=0.15,
            detector_n=128,
            detector_fov=np.deg2rad(3./60.),
            init_b=10.,
            delayline_range=1.,
            periscope_fov=np.deg2rad(5.)
    ):
        super(SIM, self).__init__()
        # photon collectors
        pc0 = PhotonCollector(d=collector_d, f=collector_f, r=collector_r, fov=optics_fov)
        pc1 = PhotonCollector(d=collector_d, f=collector_f, r=collector_r, fov=optics_fov)
        pc0.parts[0].name='E0'
        pc0.parts[1].name='M00'
        pc0.parts[2].name='M10'
        pc0.parts[3].name='M20'
        pc1.parts[0].name='E1'
        pc1.parts[1].name='M01'
        pc1.parts[2].name='M11'
        pc1.parts[3].name='M21'
        # combiner end periscope mirror
        m3_d = pc0.parts[-1].d_out
        # primary mirror of combiner
        m6_d_out = combiner_b+m3_d+2.*np.tan(.5*optics_fov)*combiner_f
        # beam waist diameter
        beam_d = m6_d_out / combiner_r
        # secondary mirror of combiner
        m7_f = combiner_f / combiner_r
        m7_g = -0.5*detector_a/np.tan(.5*detector_fov*collector_r*combiner_r)
        m6_d_in = beam_d*m7_g/(m7_g-combiner_f+m7_f) + \
            2.*np.tan(.5*optics_fov)*collector_r*combiner_r*combiner_f
        m7_d = max(
            beam_d+2.*np.tan(.5*optics_fov)*m7_f,
            m6_d_in+2.*np.tan(.5*optics_fov)*combiner_f)
        m30 = SymmetricQuadricMirror(0., m3_d, f=np.inf, g=np.inf, b=(1,0), p=[-m6_d_out*.5-.5*m3_d, 0., -combiner_f-m3_d], q=quat.from_angles(0.,  np.pi/4.), name='M30')
        m31 = SymmetricQuadricMirror(0., m3_d, f=np.inf, g=np.inf, b=(1,0), p=[ m6_d_out*.5+.5*m3_d, 0., -combiner_f-m3_d], q=quat.from_angles(0., -np.pi/4.), name='M31')
        m40 = SymmetricQuadricMirror(0., m3_d, f=np.inf, g=np.inf, b=(0,1), p=[-m6_d_out*.5-.5*m3_d, 0., 0.], q=quat.from_angles(0.,  np.pi/4.), name='M40')
        m41 = SymmetricQuadricMirror(0., m3_d, f=np.inf, g=np.inf, b=(0,1), p=[ m6_d_out*.5+.5*m3_d, 0., 0.], q=quat.from_angles(0., -np.pi/4.), name='M41')
        m50 = SymmetricQuadricMirror(0., m3_d, f=np.inf, g=np.inf, b=(0,1), p=[-combiner_b*.5, 0., 0.], q=quat.from_angles(0., -np.pi/4.), name='M50')
        m51 = SymmetricQuadricMirror(0., m3_d, f=np.inf, g=np.inf, b=(0,1), p=[ combiner_b*.5, 0., 0.], q=quat.from_angles(0.,  np.pi/4.), name='M51')
        m6  = SymmetricQuadricMirror(m6_d_in, m6_d_out, f=combiner_f, g=np.inf, b=(1,0), is_primary=True, name='M6')
        m7  = SymmetricQuadricMirror(0., m7_d, f=m7_f, g=m7_g, b=(0,1), name='M7')
        d0  = Detector(a=detector_a, N=detector_n, p=[0., 0., m7_g-m7_f], name='D0')
        # periscope between combiner and collector 0
        pr0 = OpticalSystem()
        # periscope between combiner and collector 1
        pr1 = OpticalSystem()
        # delay line controller after periscope 0
        dl0 = OpticalSystem()
        # delay line controller after periscope 1
        dl1 = OpticalSystem()
        # left arm, including photon collector 0, periscope 0, and delay line controller 0.
        arm0 = OpticalSystem()
        # right arm
        arm1 = OpticalSystem()
        # beam combiner
        bc  = OpticalSystem()
        pr0.add_part(pc0.parts[-1])
        pr0.add_part(m30)
        pr1.add_part(pc1.parts[-1])
        pr1.add_part(m31)
        dl0.add_part(m40)
        dl0.add_part(m50)
        dl1.add_part(m41)
        dl1.add_part(m51)
        bc.add_part(m30)
        bc.add_part(m31)
        bc.add_part(m6)
        bc.add_part(m7)
        bc.add_part(d0)
        bc.join(dl0)
        bc.join(dl1)
        pc0.move([-init_b*.5, 0., 0.])
        pc1.rotate([0., 0., 0., 1.])
        pc1.move([ init_b*.5, 0., 0.])
        arm0.join(pc0)
        arm0.join(pr0)
        arm0.join(dl0)
        arm1.join(pc1)
        arm1.join(pr1)
        arm1.join(dl1)
        self.join(pc0)
        self.join(pc1)
        self.join(bc)
        self.arms = [arm0, arm1]
        self.collectors = [pc0, pc1]
        self.periscopes = [pr0, pr1]
        self.delaylines = [dl0, dl1]
        self.combiner   = bc
        # set base points of sub-systems.
        # delay line controllers
        for dl in self.delaylines:
            dl.p = .5*(dl.parts[0].p + dl.parts[1].p)
        
        
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
    
