"""Optical system modelling and analysis functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import healpy as hp
from pymath.common import norm
import pymath.quaternion as quat

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
    r     - reflectance, in (z+, z-)
    f=0
    g=0
    
    2. Parabolic mirror
    focus of mirror is at origin.
    mirror opens upward (+z direction).
    Equation: f(x,y,z) = x^2 + y^2 - 4f z - 4f^2 = 0.
    Parameters:
    d_in  - inside diameter
    d_out - outside diameter
    r     - reflectance, in (z+, z-)
    f     - focal length
    g=0
    
    3. Hyperbolic mirror
    focus on the same side is at origin.
    mirror opens upward (+z direction).
    Equation: f(x,y,z) = x^2 + y^2 + (a^2-c^2)/a^2 z^2 + 2(a^2-c^2)c/a^2 z - (a^2-c^2)^2/a^2 = 0,
    where a = (g-f)/2 and c = (g+f)/2.
    Parameters:
    d_in  - inside diameter
    d_out - outside diameter
    r     - reflectance, in (z+, z-)
    f     - focal length, distance between mirror center and focus on the same side
    g     - focal length, distance between mirror center and focus on the other side
    """
    def __init__(self, d_in, d_out, r=(1., 1.), f=0., g=0., p=[0., 0., 0.], q=[1., 0., 0., 0.]):
        """Construct symmetric quadric mirror object.

        Arguments:
        d_in  - inside diameter.
        d_out - outside diameter.
        r     - reflectance, in (z+, z-), i.e., reflectance of top and bottom surface of the mirror.
        f     - focal length, for parabolic and hyperbolic mirrors.
        g     - distance to the other focus, for hyperbolic mirror only.
        p     - position in lab coordinate system.
        q     - attitude quaternion to convert coordinate of mirror's fixed csys to lab csys.
                example: r' = qrq' + p,
                where r is coordinate of mirror's fixed csys, r' is coordinate of lab csys,
                p is position of the mirror in lab csys and q is the attitude quaternion.
        """
        self.d_in    = d_in
        self.d_out   = d_out
        self.reflect = r
        self.f       = f
        self.g       = g
        self.p       = p
        self.q       = q
        if np.isclose(self.f, 0.):
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
        elif np.isclose(self.g, 0.):
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
        else:
            a = (self.g-self.f)*.5
            c = (self.g+self.f)*.5
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
        theta_out = np.arcsin(.5527708)
        theta_in  = np.arcsin(.5527708*self.d_in/self.d_out)
        npix_start, npix_stop = hp.ang2pix(nside, [theta_in, theta_out], [0., np.pi*2.])
        xs, ys, _ = np.double(hp.pix2vec(nside, range(npix_start, npix_stop)))*self.d_out
        rs = (xs**2.+ys**2.)**.5
        if np.isclose(self.f, 0.):
            zs = np.zeros_like(xs)
        elif np.isclose(self.g, 0.):
            zs = -(self.coef['x2']*xs**2. + self.coef['y2']*ys**2. + self.coef['c']) / self.coef['z']
        else:
            a = self.coef['z2']
            b = self.coef['z']
            c = self.coef['x2']*xs**2. + self.coef['y2']*ys**2. + self.coef['c']
            zs = (-b-(b**2.-4.*a*c)**.5)/2./a
        tri = mtri.Triangulation(xs, ys)
        xc = xs[tri.triangles].mean(axis=1)
        yc = ys[tri.triangles].mean(axis=1)
        rc = (xc**2.+yc**2.)**.5
        mask = np.where((rc>rs.max()) | (rc<rs.min()), 1, 0)
        tri.set_mask(mask)
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
    
    def __call__(self):
        return intersection, reflected_direction

class OpticalAssembly(object):
    """Assembly of optical parts.
    """
    def __init__(self, p=[0., 0., 0.], q=[1., 0., 0., 0.]):
        """Create empty optical assembly.

        Arguments:
        p  - position in lab coordinate system.
        q  - attitude quaternion in lab coordinate system.

        Example:
        r' = qrq' + p,
        where r is vector in assembly fixed csys and r' is vector in lab csys.
        """
        self.p = p
        self.q = q
        self.mirrors = []
    def add_mirror(self, mirror):
        self.mirrors.append(mirror)
    def move(self, dp):
        """Move assembly in lab csys by displacement vector dp.

        new position: p' = p + dp
        """
        for m in self.mirrors:
            m.p = m.p + dp
        self.p = self.p + dp
        return self.p
    def rotate(self, dq):
        """Rotate assembly around its fixed csys origin by dq.

        new attitude: q' = dq q dq'
        """
        for m in self.mirrors:
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
    def draw(self, nside=32, axes=None, return_only=False, **kwargs):
        """Draw triangular surface of all mirrors.

        Arguments:
        nside  - nside of healpix sampling points for each mirror.
        axes   - matplotlib axes object.
        return_only - instead of plotting the surface, return the triangles only.
        kwargs - keyword arguments to pass to plot_trisurf().
        """
        trigs = []
        Zs    = []
        xmin  = 0.
        xmax  = 0.
        ymin  = 0.
        ymax  = 0.
        zmin  = 0.
        zmax  = 0.
        for m in self.mirrors:
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
            if axes is None:
                fig = plt.figure()
                axes = fig.add_subplot(111, projection='3d')
            for i in range(len(self.mirrors)):
                axes.plot_trisurf(trigs[i], Zs[i], **kwargs)
            axes.set_xlim(xc-.6*sz, xc+.6*sz)
            axes.set_ylim(yc-.6*sz, yc+.6*sz)
            axes.set_zlim(zc-.6*sz, zc+.6*sz)
            plt.show()
        return trigs, Zs, extent
        
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
    
