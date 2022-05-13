"""Optical system modelling and analysis functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import healpy as hp
from pymath.common import norm
import pymath.quaternion as quat

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
        self.direction  = np.double([np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), np.cos(self.theta)]) # unit vector from the origin to the source
    def __call__(self, list_of_apertures, num_super_photons, dt):
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
                rho = .5*(((aperture.d_out**2.-aperture.d_in**2.)*np.random.rand(num_super_photons)+
                           aperture.d_in**2.)**.5)
                phi = 2.*np.pi*np.random.rand(num_super_photons)
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
                    sin_alpha = np.linalg.norm(np.cross(self.direction, norm_aperture, axis=0))
                    if sin_alpha >= (r/R):
                        beta = np.arccos(((R**2.-r**2.)**.5) / R)
                    else:
                        beta = np.arccos((R-r*sin_alpha) / (R**2.+r**2.-2.*R*r*sin_alpha)**.5)
                    rho = R*((1.-(1.-np.cos(beta))*np.random.rand(num_super_photons))**-2. - 1.)**.5
                    phi = 2.*np.pi*np.random.rand(num_super_photons)
                    s = quat.rotate(quat.from_angles(*(quat.xyz2ptr(
                        aperture.p[0]-p[0],
                        aperture.p[1]-p[1],
                        aperture.p[2]-p[2])[:2])),[R, rho*np.cos(phi), rho*np.sin(phi)])
                    u.append(s)
                    area += 2.*np.pi*(1.-np.cos(beta))*(self.rho**2.)
                u = np.concatenate(u, axis=1)
                sp_start = np.zeros((u.shape[1],), dtype=sptype)
                sp_start['weight']    = self.intensity*dt*area/self.energy/u.shape[1]
                sp_start['direction'] = quat.direction(u).transpose()
            else:
                theta = np.arccos(1.-2.*np.random.rand(num_super_photons))
                phi   = 2.*np.pi*np.random.rand(num_super_photons)
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
            sp_stop['position'][:] = np.where((t<sp_dist).reshape(-1,1), n.transpose(), sp_stop['position'])
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
            name='',
            is_primary=False,
            is_virtual=False
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
        is_primary and is_virtual are reserved switches for OpticalAssembly object.
        """
        self.d_in     = d_in
        self.d_out    = d_out
        self.boundary = b
        self.f        = f
        self.g        = g
        self.p        = np.double(p).reshape((3,1))
        self.q        = np.double(q).reshape((4,1))
        self.name     = name
        self.is_primary = is_primary
        self.is_virtual = is_virtual
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
        npix_stop = hp.ang2pix(nside, theta_out, np.pi*2.)
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
    
    def intersect(self, photon_in):
        """Find intersection of incident light ray and the mirror surface.

        Arguments:
        photon_in - incident photons

        Returns:
        intersection - coordinate of the intersection, or (np.nan, np.nan, np.nan) if the intersection
                       does not exist.
        distance     - distance between the starting point and the intersection, or np.inf if the
                       intersection does not exist.
        """
        # convert lab coordinates to mirror fixed coordinates.
        s,u = np.broadcast_arrays(
            quat.rotate(quat.conjugate(self.q), photon_in['position'].transpose()-self.p),
            quat.rotate(quat.conjugate(self.q), photon_in['direction'].transpose()))
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
        # case 3.3.2: the first intersection hits the mirror.
        r1 = np.sum((s[:2,(~a_is_0)&(~d_lt_0)][:2,tp_gt_0_and_tm_gt_0]+
                     u[:2,(~a_is_0)&(~d_lt_0)][:2,tp_gt_0_and_tm_gt_0]*t1)**2., axis=0)**.5
        # merge case 3.3.1 and 3.3.2
        tpm[tp_gt_0_and_tm_gt_0] = np.where((r1<=(self.d_out*.5))&(r1>=(self.d_in*.5)), t1, t2)
        # merge case 3.1, 3.2 and 3.3
        t[(~a_is_0)&(~d_lt_0)] = tpm[:]
        n = np.empty_like(s)
        t_is_inf = np.isinf(t)
        n[:, t_is_inf] = np.nan
        n[:,~t_is_inf] = s[:,~t_is_inf] + u[:,~t_is_inf]*t[~t_is_inf]
        rn = (n[0,~t_is_inf]**2. + n[1,~t_is_inf]**2.)**.5
        # double check out-of-boundary intersections.
        inside = np.logical_and(rn<=(self.d_out*.5), rn>=(self.d_in*.5))
        t[~t_is_inf] = np.where(inside, t[~t_is_inf], np.inf)
        n[:,~t_is_inf] = np.where(inside, n[:,~t_is_inf], np.nan)
        hit = np.logical_not(np.isinf(t))
        # convert mirror fixed coordinates back to lab coordinates.
        n[:,hit] = quat.rotate(self.q, n[:,hit]) + self.p
        return n, t

    def normal(self, r):
        """Normal vector at position r of the surface.
        A normal vector is a unit vector from given position at the surface towards the focus,
        i.e., along +z direction and perpendicular to the tangential plane of the surface.

        Argument:
        r   - coordinate in lab csys.

        Return:
        n   - unit normal vector in lab csys.
        """

        # convert r to mirror fixed coordinates
        r = quat.rotate(quat.conjugate(self.q), r-self.p)
        n = np.double([
            self.coef['x2']*2.*r[0]+self.coef['xy']*r[1]+self.coef['xz']*r[2]+self.coef['x'],
            self.coef['y2']*2.*r[1]+self.coef['xy']*r[0]+self.coef['yz']*r[2]+self.coef['y'],
            self.coef['z2']*2.*r[2]+self.coef['yz']*r[1]+self.coef['xz']*r[0]+self.coef['z']])
        if np.isinf(self.f):
            n = np.where(n[2]>0., n, -n)
        else:
            n = np.where(np.sum(n*r, axis=0)>0., n, -n)
        # convert n to lab csys
        n = quat.rotate(self.q, n)
        return n
    
    def encounter(self, photon_in, intersection):
        """Determine outcomes when a super photon encounters with the mirror.

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
        intersection - corresponding intersections in lab csys.

        Return:
        photon_out   - outcome super photons, with position and direction vectors in lab csys.
        """
        photon_out = np.empty_like(photon_in)
        n = self.normal(intersection)        # normal vector in lab csys
        phi_n, theta_n, _ = quat.xyz2ptr(*n) # longtitude and latitude of normal vector in lab csys
        # attitude Euler's angles
        phi_a   = np.where(theta_n>0., phi_n-np.pi, phi_n)
        theta_a = np.where(theta_n>0., np.pi/2.-theta_n, np.pi/2.+theta_n)
        q = quat.from_angles(phi_a, theta_a)
        photon_out['weight'][:]     = photon_in['weight']
        photon_out['wavelength'][:] = photon_in['wavelength']
        photon_out['position'][:]   = intersection.transpose()
        p = np.transpose(photon_out['position'] - photon_in['position'])
        u = quat.rotate(quat.conjugate(q), photon_in['direction'].transpose())
        d = np.reshape(p[0]*n[0] + p[1]*n[1] + p[2]*n[2], (-1, 1))
        top_mask = np.double([np.abs(self.boundary[0]), np.abs(self.boundary[0]), -self.boundary[0]])
        bot_mask = np.double([np.abs(self.boundary[1]), np.abs(self.boundary[1]), -self.boundary[1]])
        v = u.transpose()*np.where(d>0., top_mask, bot_mask)
        photon_out['direction'][:] = quat.rotate(q, v.transpose()).transpose()
        photon_out['phase'][:]     = photon_in['phase'] + \
            2.*np.pi*np.sum(p.transpose()**2.,axis=1)**.5/photon_in['wavelength']
        photon_out['position'][:] += dm*photon_out['direction']
        photon_out['phase'][:]     = np.mod(photon_out['phase'][:]+2.*np.pi*ou, 2.*np.pi)
        return photon_out

class OpticalAssembly(object):
    """Assembly of optical parts.
    """
    def __init__(self, p=[0., 0., 0.], q=[1., 0., 0., 0.], name=''):
        """Create empty optical assembly.

        Arguments:
        p    - position in lab coordinate system.
        q    - attitude quaternion in lab coordinate system.
        name - human readable name assigned to the assembly.

        Example:
        r' = qrq' + p,
        where r is vector in assembly fixed csys and r' is vector in lab csys.
        """
        self.p = p
        self.q = q
        self.parts = []
        self.name = name
        
    def add_part(self, mirror):
        self.parts.append(mirror)
        
    def move(self, dp):
        """Move assembly in lab csys by displacement vector dp.

        new position: p' = p + dp
        """
        for m in self.parts:
            m.p = m.p + dp
        self.p = self.p + dp
        return self.p
    
    def rotate(self, dq):
        """Rotate assembly around its fixed csys origin by dq.

        new attitude: q' = dq q dq'
        """
        for m in self.parts:
            m.p = quat.rotate(dq, m.p)
            m.q = quat.multiply(dq, m.q)
        self.q = quat.multiply(dq, self.q)
        return self.q
    
    def set_p(self, p):
        """Move to new position p.
        """
        dp = p-self.p
        self.move(dp)
        
    def set_q(self, q):
        """Rotate to new attitude q.
        """
        dq = quat.multiply(q, quat.conjugate(self.q))
        self.rotate(dq)
        
    def draw(
            self,
            nside=32,
            axes=None,
            return_only=False,
            draw_virtual=True,
            highlight_primary=True,
            highlight_list=[],
            raytrace=None,
            virtual_opts={},
            highlight_opts={},
            surface_opts={},
            lightray_opts={}
    ):
        """Draw triangular surface of all mirrors.

        Arguments:
        nside              - nside of healpix sampling points for each mirror.
        axes               - matplotlib axes object.
        return_only        - instead of plotting the surface, return the triangles only (boolean switch).
        draw_virtual       - draw virtual mirrors (boolean switch).
        highlight_primary  - highlight primary mirrors (boolean switch).
        highlight_list     - list of indices of highlight mirrors.
        raytrace           - plot ray trace (optional).
        virtual_opts       - plot_trisurf keyword options for virtual surfaces, e.g., entrance pupil.
        highlight_opts     - plot_trisurf keyword options for highlighted surfaces, e.g., primary mirror.
        surface_opts       - plot_trisurf keyword options for ordinary surfaces.
        lightray_opts      - plot keyword options for light rays.
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
        if not return_only:
            if len(virtual_opts)==0:
                virtual_opts={
                    'alpha': 0.6,
                    'color': 'lightcyan',
                    'linestyle': 'None'
                }
            if len(highlight_opts)==0:
                highlight_opts={
                    'alpha': 1.0,
                    'color': 'red',
                    'linestyle': 'None'
                }
            if len(lightray_opts)==0:
                lightray_opts={
                    'alpha':     0.5,
                    'linewidth': 0.1,
                    'color':     'green'
                }
            if len(surface_opts)==0:
                surface_opts={
                    'alpha': 1.0,
                    'color': 'darkgoldenrod',
                    'linestyle': 'None'
                }
            if axes is None:
                fig = plt.figure()
                axes = fig.add_subplot(111, projection='3d')
            for i in range(len(self.parts)):
                if self.parts[i].is_virtual:
                    if draw_virtual:
                        axes.plot_trisurf(trigs[i], Zs[i], **virtual_opts)
                    continue
                if (i in highlight_list) or (highlight_primary and self.parts[i].is_primary):
                    axes.plot_trisurf(trigs[i], Zs[i], **highlight_opts)
                    continue
                axes.plot_trisurf(trigs[i], Zs[i], **surface_opts)
            if raytrace is not None:
                nodes, npts = raytrace.shape
                for i in range(npts):
                    axes.plot(
                        raytrace['position'][:,i,0],
                        raytrace['position'][:,i,1],
                        raytrace['position'][:,i,2],
                        **lightray_opts
                    )
            axes.set_xlim(xc-.6*sz, xc+.6*sz)
            axes.set_ylim(yc-.6*sz, yc+.6*sz)
            axes.set_zlim(zc-.6*sz, zc+.6*sz)
            plt.show()
        return trigs, Zs, extent
    
    def intersect(self, photon_in):
        """Find intersections of incident light rays and the mirrors.

        Arguments:
        photon_in - incident super photons.

        Returns:
        n - intersections, in lab csys.
        t - optical path lengths between starting points and intersections.
        k - indices of mirror encounted with.
        """
        n = np.empty((3, photon_in.size), dtype='double')
        t = np.empty((photon_in.size,  ), dtype='double')
        k = np.empty((photon_in.size,  ), dtype=mstype  )
        n[:] = np.nan
        t[:] = np.inf
        k[:] = -1
        for i in range(len(self.parts)):
            ni, ti = self.parts[i].intersect(photon_in)
            m = np.bool_(ti<t)
            n[:] = np.where(m, ni, n)
            t[:] = np.where(m, ti, t)
            k[:] = np.where(m,  i, k)
        return n, t, k

    def trace(self, photon_in, steps=1):
        """Trace photons in optical assembly.

        Arguments:
        photon_in  - input photons
        steps      - number of ray-tracing steps

        Returns:
        photon_trace - photon trace at each step
        """
        photon_trace = np.empty((steps+1, photon_in.size), dtype=sptype)
        mirror_sequence = np.empty((steps+1, photon_in.size), dtype=mstype)
        photon_trace[0, :] = photon_in[:]
        mirror_sequence[0, :] = -1
        for i in range(steps):
            n,t,k = self.intersect(photon_trace[i, :])
            mirror_sequence[i+1,:] = k
            m = np.isinf(t)
            photon_trace[i+1, m] = photon_trace[i,m]
            hits = np.bincount(k[~m],minlength=len(self.parts))
            for l in range(len(self.parts)):
                if hits[l]>0:
                    m = np.bool_(k==l)
                    photon_trace[i+1, m] = self.parts[l].encounter(photon_trace[i, m], n[:, m])
        return photon_trace, mirror_sequence
        
        
class PhotonCollector(OpticalAssembly):
    pass

class SIM(OpticalAssembly):
    """Simplified Michelson stellar interferometer model.

1. Overall model

      M10                                      M11
 |    --   |                              |    --    |
 |   /| \  |                              |   / |\   |
 |  / ||---+------------- B --------------+----|| \  |
 | /  |   \|                              | /   |  \ |
 ---- |  ---- M00                     M01 ----  |  ----
      |              |--- B' ---|               |
      |                  ____ M5                |
      \--------------\   /  \   /-------------- /
     M20         M30 |  /|  |\  | M31            M21
                     | / |  | \ |
                     |/  |  |  \|
                    ---- \  / ----- M4
                          ||
                          ||
                          \/
                          -- D0

M00 (M01) - Concave parabolic mirror, primary mirror of left (right) collector.
M10 (M11) - Convex parabolic mirror, secondary mirror of left (right) collector.
M20 (M21) - Plane mirror, periscope mirror of left (right) collector.
M30, M31  - Plane mirrors, periscope mirrors of beam combiner.
M4        - Concave parabolic mirror, primary mirror of beam combiner.
M5        - Convex hyperbolic mirror, secondary mirror of beam combiner.
D0        - Pixel detector array.

2. Structual models
2.1 Left collector
The left collector contains M00, M10 and M20.
M20 has tip and tilt actuators.
2.2 Right collector
The right collector contains M01, M11 and M21.
M21 also has tip and tilt actuators.
2.3 Beam combiner
The beam combiner contains M30, M31, M4, M5 and D0.
M30 and M31 both have tip and tilt actuators.
2.4 Degrees of freedom
Each structure has 3 translational as well as 3 rotational degrees of freedom.

3. Optical models
3.1 Photon collecting telescopes
M00 (M01) and M10 (M11) constitute the left (right) collecting telescope.
3.2 Periscopes
M20 (M21) and M30 (M31) constitute the left (right) periscope.
3.3 Beam combining telescope
M4 and M5 constitute the beam combining telescope.

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
    def __init__(self):
        pass
    def __call__(self):
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
    
