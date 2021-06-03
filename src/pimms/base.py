"""This is the base module of PIMMS package. It provides a series of
basic utilities for other modules of the package.
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path
from scipy import misc
from scipy.interpolate import interp2d
from scipy.fftpack import fftshift,ifftshift,fftn,fft2,ifftn
from scipy.special import j1
from pymath.quadrature import *

DEPS = np.finfo(np.double).eps

def rgb_to_hsl(r,g,b):
    """Convert RGB color space components to HSL color space components.
Reference: http://en.wikipedia.org/wiki/HSL_and_HSV
"""
    R, G, B = np.double([r,g,b]) / np.max([r,g,b])
    maximum = np.max([R, G, B], axis=0)
    minimum = np.min([R, G, B], axis=0)
    chroma  = maximum - minimum
    H  = np.zeros(R.shape, dtype=R.dtype)
    RM = np.logical_and(chroma>0, np.isclose(maximum, R))
    GM = np.logical_and(chroma>0, np.isclose(maximum, G))
    BM = np.logical_and(chroma>0, np.isclose(maximum, B))
    H[RM] = np.mod((G[RM] - B[RM])/chroma[RM], 6.)
    H[GM] = (B[GM] - R[GM])/chroma[GM] + 2.
    H[BM] = (R[BM] - G[BM])/chroma[BM] + 4.
    H = H / 6.
    L = 0.5*(maximum+minimum)
    S = np.zeros(R.shape, dtype=R.dtype)
    LM = np.logical_and(L>0., L<1.)
    S[LM] = chroma[LM] / (1. - np.abs(2.*L[LM]-1))
    S = S/S.max()
    return H, S, L

def hsl_to_rgb(h, s, l):
    CX0 = np.zeros((3,) + h.shape, dtype=h.dtype)
    CX0[0,:] = s * (1. - np.abs(2.*l - 1.))
    CX0[1,:] = CX0[0,:] * (1. - np.abs(np.mod(h*6., 2.) - 1.))
    R = np.zeros(h.shape, dtype=h.dtype)
    G = np.zeros(h.shape, dtype=h.dtype)
    B = np.zeros(h.shape, dtype=h.dtype)
    selector = [
        [0,1,2],
        [1,0,2],
        [2,0,1],
        [2,1,0],
        [1,2,0],
        [0,2,1]
    ]
    for i in range(6):
        m = np.logical_and(h>=(i/6.), h<=(i+1.)/6.)
        R[m], G[m], B[m] = CX0[selector[i][0]][m], CX0[selector[i][1]][m], CX0[selector[i][2]][m]
    m = l - CX0[0,:]/2.
    R, G, B = R+m, G+m, B+m
    return R, G, B

def histnorm(x):
    """Normalize color space using histogram-related remapping.

Input:
x     - N-D array.
"""
    p = np.argsort(x.ravel())
    y = x.ravel()[p]
    d = np.zeros_like(y)
    d[1:] = np.diff(y)
    s = np.cumsum(d>0)
    q = np.zeros(s.shape, dtype='float64')
    q[p] = s[:]
    q_min = np.min(q)
    q_max = np.max(q)
    q = (q-q_min) / max(1.0, q_max-q_min)
    return np.reshape(q, x.shape)
    
def airydisk(
        loc        = (0.0,0.0),
        pxsz       = (11.5,11.5),
        size       = (16,16),
        fnumber    =  2.0,
        wavelength =  550.0
):
    """Airy disk, non-normalized.

loc        - (x,y) is the location of the center of the Airy disk with respect to
             the reference point of the pixel grid.
             The reference point of the pixel grid is at the geometrical
             center.
pxsz       - (a,b) is the size of one pixel. a is the length of the side
             of a pixel along the x-axis (row) while b is the length of the
             side of it along the y-axis (column). Both are in microns.
size       - (M,N) is the size of the pixel grid. M is the number of rows while
             N is the number of columns.
fnumber    - f-number of the optics.
wavelength - wavelength in nano-metres.
"""
    nzs = np.min(pxsz)*1.0e-6 / (fnumber * wavelength * 1.0e-9) # periods per pixel

    if nzs >= 16:
        a = GLQA64
        w = GLQW64
    elif nzs >= 8:
        a = GLQA32
        w = GLQW32
    elif nzs >= 4:
        a = GLQA16
        w = GLQW16
    elif nzs >= 2:
        a = GLQA8
        w = GLQW8
    else:
        a = GLQA4
        w = GLQW4
    if size is None:
        size = (max(16,int(20.0/nzs+2.0*np.abs(loc[1]))), max(16,int(20.0/nzs+2.0*np.abs(loc[0]))))
    psf = lambda x,y: (2.0*j1(np.pi*((x-loc[0]*pxsz[0]*1.0e-6)**2.0 + (y-loc[1]*pxsz[1]*1.0e-6)**2.0)**0.5 /
                     (wavelength*1.0e-9) / fnumber) *
                     (wavelength*1.0e-9*fnumber) /
                     (np.pi * ((x-loc[0]*pxsz[0]*1.0e-6)**2.0 + (y-loc[1]*pxsz[1]*1.0e-6)**2.0)**0.5))**2.0
    W      = np.array(np.matrix(w.reshape((-1,1))) * np.matrix(w.reshape(1,-1)))
    Xa, Ya = np.meshgrid(a, a, indexing='xy')
    w  = W.ravel()
    xa = Xa.ravel()
    ya = Ya.ravel()
    xl = (np.arange(size[1])   - size[1]*0.5) * pxsz[1] * 1.0e-6
    xu = (np.arange(size[1])+1 - size[1]*0.5) * pxsz[1] * 1.0e-6
    yl = (np.arange(size[0])   - size[0]*0.5) * pxsz[0] * 1.0e-6
    yu = (np.arange(size[0])+1 - size[0]*0.5) * pxsz[0] * 1.0e-6
    X0, Y0, Xa = np.meshgrid(xl, yl, xa, indexing='xy')
    _ , _ , Ya = np.meshgrid(xl, yl, ya, indexing='xy')
    _ , Y1, W  = np.meshgrid(xl, yu, w,  indexing='xy')
    X1, _ , _  = np.meshgrid(xu, yu, xa, indexing='xy')
    return np.sum(W*psf(pxsz[0]*0.5e-6*Xa+(X0+X1)*0.5, pxsz[1]*0.5e-6*Ya+(Y0+Y1)*0.5), axis=2)

def imresize(f,ratio,kind='linear'):
    w,h = f.shape
    x  = np.arange(1.0*w)/np.arange(w).max()
    y  = np.arange(1.0*h)/np.arange(h).max()
    xi = np.arange(1.0*w*ratio)/np.arange(1.0*w*ratio).max()
    yi = np.arange(1.0*h*ratio)/np.arange(1.0*h*ratio).max()
    g  = interp2d(x,y,f,kind=kind)
    return g(xi,yi)

def dlt(f,order=2):
    '''DLT discrete Laplace transform of 2D array
    '''
    if order < 4:
        h = [[0, 1, 0],[1, -4, 1],[0, 1, 0]]
    elif order < 6:
        h = [[0,       0,     -1.0/12,  0,      0],\
             [0,       0,      4.0/3,   0,      0],\
             [-1.0/12, 4.0/3, -5.0,     4.0/3, -1.0/12],\
             [0,       0,      4.0/3,   0,      0],\
             [0,       0,     -1.0/1,   0,      0]]
    else:
        h = [[   0,     0,   0,   1.0/90,   0,     0,    0],\
             [   0,     0,   0,  -3.0/20,   0,     0,    0],\
             [   0,     0,   0,    3.0/2,   0,     0,    0],\
             [1.0/90, -3.0/20, 3.0/2,  -49.0/9, 3.0/2, -3.0/20, 1.0/90],\
             [   0,     0,   0,    3.0/2,   0,     0,    0],\
             [   0,     0,   0,  -3.0/20,   0,     0,    0],\
             [   0,     0,   0,   1.0/90,   0,     0,    0]];
    return imconv(f,h,'symmetric')

def shift_to_src_slice(sft):
    if sft>0:
        return slice(0,-sft)
    else:
        return slice(-sft,None)

def shift_to_dest_slice(sft):
    if sft>=0:
        return slice(sft,None)
    else:
        return slice(0,sft)

def zeroshift(X,sft):
    """N-D array shift with zero boundary.
    X is the input N-D array.
    sft is specified as (i,j,...), where i,j,... are shifts along the 0-axis,
    1-axis,... respectively.
    """
    Xsz = np.shape(X)
    Xdim = np.size(Xsz)
    if np.size(sft) == 1:
        X = np.ravel(X)
        Y = np.zeros(np.size(X))
        ss = shift_to_src_slice(sft)
        sd = shift_to_dest_slice(sft)
        Y[sd] = X[ss]
        return np.reshape(Y,Xsz)
    else:
        if (Xdim != np.size(sft)) & (np.size(sft)>1):
            raise StandardError('The dimension of the input array does not\
                agree with the length of the shift vector.')
        Y = np.zeros(Xsz)
        ss = map(shift_to_src_slice, sft)
        sd = map(shift_to_dest_slice, sft)
        Y[sd] = X[ss]
        return Y

# Standard deviation of multiresolution components of 1D white noise.
# Standard deviation of multiresolution components of 2D white noise.
# Standard deviation of multiresolution components of 3D white noise.
MR_NORMAL_STD1 = [0.700, 0.323, 0.210, 0.141, 0.099, 0.071, 0.054]
MR_NORMAL_STD2 = [\
    0.890544, 0.200866, 0.085688, 0.041390, 0.020598, \
    0.010361, 0.005256, 0.002706, 0.001412, 0.000784];
MR_NORMAL_STD3 = [0.956, 0.120, 0.035, 0.012, 0.004, 0.001, 0.0005]

def ianscombe(g):
    """Inverse Anscombe transform.
    """
    return (g/2.0)**2.0 - 3.0/8.0

def anscombe(x):
    """Anscombe transform.
    """
    return 2.0*np.sqrt(x + 3.0/8.0)

def auto_noise_estimate(X,J=None,k=3,epsilon=1e-3):
    """Automated noise estimation from the multiresolution support.
    Reference:
      Starck, JL, Astronomical image and data analysis, 2006.
    """
    sigma = []
    sigma.append(k_sigma_clipping(X,k))
    C,W = dwt(X,J)
    _,_,J = np.shape(W)
    n = 0
    delta = 1
    while delta > epsilon:
        M = np.array(np.abs(W)>=np.reshape(\
            k*sigma[n]*np.array(MR_NORMAL_STD2)[0:J],(1,1,J)),dtype=bool)
        S = np.sum(M,axis=2)
        if np.all(S):
            break
        else:
            sigma.append(k_sigma_clipping(np.ma.masked_array(X-C,S),k))
        n += 1
        delta = np.abs(sigma[n]-sigma[n-1])/sigma[n]
        print("iteration %d, delta = %f"%(n,delta))
    return sigma[n]/0.974

def k_sigma_clipping(X,k=3):
    """Estimate standard deviation of Gaussian noise in data X.
    Reference:
      Starck, JL, Astronomical image and data analysis, 2006.
    """
    MAX_LOOP = 500
    if np.ma.is_masked(X):
        XM = X.mask
    else:
        XM = np.zeros(np.shape(X),dtype=bool)
    if np.all(XM):
        raise StandardError('All elements are masked.')
    X = X-np.mean(X)
    sigma = np.std(X)
    for n in range(0,MAX_LOOP):
        M = np.array(np.abs(X)>(k*sigma),dtype=bool) | XM
        if np.any(~M):
            #sigma_new = np.std(X[~M])
            sigma_new = np.std(np.ma.masked_array(X,M))
        else:
            return sigma
        if sigma_new >= sigma:
            return sigma_new
        sigma = sigma_new
    print('MAX_LOOP reached before convergence.')
    return sigma

def up_sample(X,J=1):
    """2D image nearest up sample.

J is number of down samples.
"""
    for j in range(J):
        N,M = np.shape(X)
        Y = np.empty((N*2,M*2),dtype=X.dtype)
        Y[::2,::2] = X
        Y[::2,1::2] = X
        Y[1::2,::2] = X
        Y[1::2,1::2] = X
        X = Y
    return X

def down_sample(X,J=None,h=None):
    """2D image down sample.

J is number of down samples.
h is samping coefficients.
Size of each axis reduced by a half by evergy down sample.
"""
    for d in range(0,J):
        X = dwt(X,J=1,h=h)[0][::2,::2]
    return X


def dwt(X,J=None,h=None):
    """2D discrete wavelet transform.

X is the 2D array.
J is the maximum scale.
h is scale coefficients for wavelet transform.
"""
    if h is None:
        h = np.double([1,4,6,4,1])/16.0
    if np.ndim(h) == 1:
        h = np.array(np.matrix(h).T * np.matrix(h))
    if J is None:
        J = np.int64(np.ceil(np.log2(np.min(np.shape(X)))))
    if J <= 0:
        return X,np.zeros(np.shape(X))
    szH = np.shape(h)
    ncoe = np.prod(szH)
    sr,sc = np.meshgrid(range(szH[0]), range(szH[1]), indexing='ij')
    sr = sr - np.round((szH[0])/2.0)
    sc = sc - np.round((szH[1])/2.0)
    C = np.array(X)
    W = np.zeros(np.shape(X)+(J,))
    for k in range(0,J):
        Cn = np.zeros(np.shape(X))
        for l in range(0,ncoe):
            Cn += symmshift(C,[sr.reshape(-1)[l],sc.reshape(-1)[l]]) * \
                h.reshape(-1)[l]
        W[:,:,k] = C - Cn;
        C = Cn
        sr,sc = sr*2,sc*2
    return C,W

def symmshift(X,sft):
    """2D array shift with symmetric boundary.

X is the input image.
sft is specified as (i,j), where i is shift along the first axis (y-axis)
and j is shift along the second axis (x-axis).
"""
    szX = np.shape(X)
    rgv,cgv = np.int64(np.arange(0,szX[0])+sft[0]),\
        np.int64(np.arange(0,szX[1])+sft[1])
    rgv = np.mod(rgv,szX[0]*2)
    cgv = np.mod(cgv,szX[1]*2)
    rrev = np.array(rgv>=szX[0],dtype=bool)
    crev = np.array(cgv>=szX[1],dtype=bool)
    rgv[rrev] = 2*szX[0]-1-rgv[rrev]
    cgv[crev] = 2*szX[1]-1-cgv[crev]
    ridx, cidx = np.meshgrid(rgv, cgv, indexing='ij')
    return np.array(X)[ridx, cidx]

def subshift(I,sft,fill_value=0.0):
    """2D array sub-pixel shift.

I is the input image.
sft is specified as (i,j), where i is shift along the first axis (y-axis)
and j is shift along the second axis (x-axis).
Outside values are filled with fill_value.
"""
    dy,dx = np.shape(I)
    xgv = np.arange(0,dx) - (dx-1)*0.5
    ygv = np.arange(0,dy) - (dy-1)*0.5
    f = interp2d(xgv,ygv,I,kind='linear',bounds_error=False,\
        fill_value=fill_value)
    return f(xgv-sft[1],ygv-sft[0])

def centroid(I):
    """Find the xy location (using Cartesian indexing) of the centroid of the

input image I. The origin of the xy coordinate is at the centre of the image
grid.
I is the input 2-D image.
Return:
cx and cy are the x-axis coordinate and y-coordinate of the centroid
respectively.
"""
    dy,dx = np.shape(I)
    xgv = np.arange(0,dx) - (dx-1)*0.5
    ygv = np.arange(0,dy) - (dy-1)*0.5
    x,y = np.meshgrid(xgv,ygv,indexing='xy')
    cx = np.sum(x*I) / np.sum(I)
    cy = np.sum(y*I) / np.sum(I)
    return cx,cy

def bootstrap(estimator,sample_data,resamples=10,resample_size=None):
    """Measure accuracy of an estimator by bootstrap method.
    estimator is a callable object.
    sample_data is original observed data, the data source for resampling.
    resamples is the number of resampled data set.
    resample_size is the size of resampled data set.
    """
    if resample_size is None:
        resample_size = np.size(sample_data)
    estimates = []
    for k in range(0,resamples):
        estimates.append(estimator(np.array(sample_data)[roulette(np.size(sample_data),\
            resample_size)]))
    return estimates

def roulette(stop,repeats=1,weights=None):
    """Generate random indices.
    """
    if weights is None:
        idx = np.int64(np.random.rand(repeats)*stop)
    else:
        weights = np.double(weights) / np.sum(weights)
        ul = np.cumsum(weights)
        ll = np.double([0]+list(ul[0:-1]))
        rnd = np.random.rand(repeats)
        idx = map(lambda r:np.nonzero((r>=ll)&(r<=ul))[0][0],rnd)
    return idx

def parse_config_file(filename):
    """Parse config file and return python dict that contains all keys and
    values defined in the file.
    Definition of config file:
    key = value
    key is case insensitive.
    comment line starts with '#' or '%'.
    """
    cdict = {}
    with open(filename,'r') as f:
        for line in f:
            if line[0] != '#' and line[0]!='%':
                dlmt = line.find('=')
                if dlmt != -1:
                    key = line[0:dlmt].lower().replace(' ','')
                    val = line[(dlmt+1):].lower().replace(' ','').\
                        replace('\n','').replace('\r','')
                    cdict[key] = val
    return cdict

def imrotate(fxy,x=None,y=None,angle=None,box='loose',method='bilinear'):
    """Rotate image by an angle around the origin of the specified coordinate
    system. The default origin is the centre of the image (but not necessarily
    any pixel of the image).
    """
    ny,nx = np.shape(fxy)
    if x is None:
        x = np.arange(0.0,nx,1.0) - (nx-1.0)*0.5
    if y is None:
        y = np.arange(0.0,ny,1.0) - (ny-1.0)*0.5
    cx = np.double([x[0],x[nx-1],x[nx-1],x[0]])
    cy = np.double([y[0],y[0],y[ny-1],y[ny-1]])
    cu = np.cos(angle)*cx-np.sin(angle)*cy
    cv = np.sin(angle)*cx+np.cos(angle)*cy
    umax = np.max(cu)
    umin = np.min(cu)
    vmax = np.max(cv)
    vmin = np.min(cv)
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    if box.lower()[0] == 'l':
        u = np.arange(umin,umax,dx)
        v = np.arange(vmin,vmax,dy)
    elif box.lower()[0] == 'c':
        u = x
        v = y
    else:
        raise StandardError('Unsupported box type '+str(box))
    U,V = np.meshgrid(u,v)
    xi,yi = (np.cos(angle)*U-np.sin(angle)*V,\
        np.sin(angle)*U+np.cos(angle)*V)
    #uv2xy = lambda x,y:(np.cos(angle)*x-np.sin(angle)*y,\
        #np.sin(angle)*x+np.cos(angle)*y)
    fuv = bilinear(x,y,fxy,xi,yi)
    return fuv,u,v

def rect2polar(fxy,x=None,y=None,rho=None,phi=None,rgv=None,pgv=None,\
        resample=4.0,extrap=0):
    """Transform image f(x,y) from rectangular grid (x,y) to polar grid
    (rho,phi).
    fxy is an image on rectangular pixel grid approximates the function
    f at a certain precision.
    x and y are coordinates of the rectangular pixel grid. If x and y are
    1-D arrays, the coordinate of (i,j) (i-th row, j-th column) is
    (x(j),y(i)). If x and y are 2-D arrays with the same shape, the
    coordinate of (i,j) is (x(i,j),y(i,j)).
    rgv is 1-D array. It is the grid variable of radius coordinate of the
    polar grid.
    pgv is 1-D array. It is the grid variable of polar angle coordinate of
    the polar grid.
    rho and phi are ndarrays of the same size. rho and phi override rgv and
    pgv.
    """
    ny,nx = np.shape(fxy)
    if x is None:
        x = np.arange(0.0,nx,1.0) - (nx-1.0)*0.5
    if y is None:
        y = np.arange(0.0,ny,1.0) - (ny-1.0)*0.5
    if np.ndim(x) == 1:
        x,y = np.meshgrid(x,y)
    xgv,ygv = x[0,:],y[:,0]
    if rho is None or phi is None:
        rxy = np.sqrt(x**2.0+y**2.0)
        pxy = np.arctan2(y,x)
        rmax,rmin = rxy.max(),rxy.min()
        pmax,pmin = pxy.max(),pxy.min()
        nxy = np.max([nx,ny])
        nrho = nxy / 2.0 * resample
        nphi = nxy * resample
        if rgv is None:
            rgv = (np.arange(0.0,nrho,1.0)/(nrho-1.0))*(rmax-rmin)+rmin
        if pgv is None:
            if (x>=0).any():
                pgv=(np.arange(0.0,nphi,1.0)/(nphi-1.0))*(pmax-pmin)+pmin
            else:
                pmin = np.mod(pmin,2.0*np.pi);
                pmax = np.mod(pmax,2.0*np.pi);
                pgv=(np.arange(0.0,nphi,1.0)/(nphi-1.0))*(pmax-pmin)+pmin
        [rho,phi] = np.meshgrid(rgv,pgv);
    fp = np.zeros(rho.shape)
    xref,yref = xgv[0],ygv[0]
    xdel,ydel = np.mean(np.diff(xgv)),np.mean(np.diff(ygv))
    xp,yp = rho*np.cos(phi),rho*np.sin(phi)
    jp = np.int64(np.round((xp-xref)/xdel))
    ip = np.int64(np.round((yp-yref)/ydel))
    msk = (ip>=0) & (ip<ny) & (jp>=0) & (jp<nx)
    fp[msk] = fxy[ip[msk],jp[msk]]
    #fp = bilinear(xgv,ygv,fxy,xp,yp,extrap=extrap)
    return fp,msk,rho,phi

def polar2rect(fp,x=None,y=None,rho=None,phi=None,\
        resample=0.25,extrap=0):
    """Transform image from polar grid to rectangular grid.
    """
    nphi,nrho = np.shape(fp)
    if phi is None and rho is None and x is None and y is None:
        nxy = max(nrho*2.0,nphi) * resample
        nx = nxy
        ny = nxy
        xgv = np.arange(0,nxy) - (nxy-1.0)*0.5
        ygv = np.arange(0,nxy) - (nxy-1.0)*0.5
        [x,y] = np.meshgrid(xgv,ygv)
        rxy = np.sqrt(x**2.0 + y**2.0)
        pxy = np.arctan2(y,x)
        rmax,rmin = rxy.max(),rxy.min()
        pmax,pmin = pxy.max(),pxy.min()
        rgv = (np.arange(0.0,nrho,1.0)/(nrho-1.0))*(rmax-rmin)+rmin
        pgv=(np.arange(0.0,nphi,1.0)/(nphi-1.0))*(pmax-pmin)+pmin
        [rho,phi] = np.meshgrid(rgv,pgv);
    if phi is None:
        phi = np.arange(0,nphi)/(nphi-1.0)*2.0*np.pi-np.pi
    if rho is None:
        rho = np.arange(0,nrho)*np.sqrt(2.0)
    if np.ndim(phi) == 1:
        rho,phi = np.meshgrid(rho,phi)
    xp,yp = rho*np.cos(phi),rho*np.sin(phi)
    if x is None or y is None:
        nxy = max(nrho*2.0,nphi) * resample
        xmax,xmin = xp.max()/np.sqrt(2.0),xp.min()/np.sqrt(2.0)
        ymax,ymin = yp.max()/np.sqrt(2.0),yp.min()/np.sqrt(2.0)
        xgv = np.arange(0,nxy)/(nxy-1.0)*(xmax-xmin)+xmin
        ygv = np.arange(0,nxy)/(nxy-1.0)*(ymax-ymin)+ymin
    else:
        if np.ndim(x) == 1:
            xgv,ygv = x,y
        else:
            xgv,ygv = x[0,:],y[:,1]
    nx,ny = np.size(xgv),np.size(ygv)
    xref,yref = xgv[0],ygv[0]
    xdel,ydel = np.mean(np.diff(xgv)),np.mean(np.diff(ygv))
    jp = np.int64(np.round((xp-xref)/xdel))
    ip = np.int64(np.round((yp-yref)/ydel))
    kp = ip*nx+jp
    msk = (ip>=0) & (ip<ny) & (jp>=0) & (jp<nx)
    rep = np.reshape(np.bincount(kp[msk].flatten(),minlength=nx*ny),(ny,nx))
    fxy = np.reshape(np.bincount(kp[msk].flatten(),fp[msk].flatten(),minlength=nx*ny),(ny,nx))
    fxy[rep>0] = fxy[rep>0] / rep[rep>0]
    return fxy,rep,xgv,ygv

def gauss(shape,sigma=None,fwhm=None):
    if sigma is None:
        sigma = fwhm / 2.3548
    if np.isscalar(shape):
        shape = (shape,shape)
    if np.isscalar(sigma):
        sigma = (sigma,sigma)
    x = np.arange(0,shape[1]) - (shape[1]-1)*0.5
    y = np.arange(0,shape[0]) - (shape[0]-1)*0.5
    x,y = np.meshgrid(x,y,indexing='xy')
    g = np.exp(-0.5*((x**2.0)/(sigma[1]**2.0) + (y**2.0)/(sigma[0]**2.0)))
    g = g / np.sum(g)
    return g

def imtrans(x,y,fxy,uv2xy,u,v):
    """Transform image from (x,y) grid to (u,v) grid.
    x and y are equidistant 1-D coordinates.
    fxy is the function defined on the (x,y) grid that represents the image.
    uv2xy is a callable object that takes (u,v) as input and maps (u,v) to
    (x,y) pixelwisely.
    u and v are equidistant 1-D coordinates (u is horizontal and v is verticle).
    """
    U,V = np.meshgrid(u,v)
    xi,yi = uv2xy(U,V)
    fuv = bilinear(x,y,fxy,xi,yi)
    return fuv

def bilinear(x,y,z,xi,yi,extrap=0):
    """Bilinear interpolation.
    x and y are 1-D arrays which define equidistant data grid.
    z is the sampled 2-D data.
    xi and yi are n-D arrays.
    Any missing data on (xi,yi) is interpolated with this function.
    """
    if len(x) == 0:
        dx = 1.0
        xn = np.int64(np.floor(xi))
        xr = xi - xn
    else:
        dx = np.mean(np.diff(x))
        xn = np.int64(np.floor((xi-x[0])/dx))
        xr = ((xi-x[0])/dx) - xn
    if len(y) == 0:
        dy = 1.0
        yn = np.int64(np.floor(yi))
        yr = yi - yn
    else:
        dy = np.mean(np.diff(y))
        yn = np.int64(np.floor((yi-y[0])/dy))
        yr = ((yi-y[0])/dy) - yn
    ymax,xmax = np.shape(z)
    zi = np.zeros(xn.shape)
    msk = (xn>=0) & (xn<(xmax-1)) & (yn>=0) & (yn<(ymax-1))
    zi[~msk] = extrap
    zi[msk] = (1.0-xr[msk])*((1.0-yr[msk])*z[yn[msk],xn[msk]] + \
        yr[msk]*z[yn[msk]+1,xn[msk]]) + \
        xr[msk]*((1.0-yr[msk])*z[yn[msk],xn[msk]+1] + \
        yr[msk]*z[yn[msk]+1,xn[msk]+1])
    msk = (xn>=0) & (xn<xmax) & (yn>=0) & (yn<ymax) & (xr<=DEPS) & (yr<=DEPS)
    zi[msk] = z[yn[msk],xn[msk]]
    return zi

def imconv(f,kernel=None,extension='circular',verbose=False):
    """Convolve an image with a kernel function.
    kernel
    extension
      'circular' (or 'cyclic', 'periodic')
      'symmetric'
      constant
    """
    # check input image and kernel sizes
    szH = np.shape(kernel)
    szF = np.shape(f)
    if verbose:
        print('Input image size: ' + str(szF))
        print('Input kernel size: ' + str(szH))
    if np.any(np.greater(szH, szF)):
        raise ValueError('Kernel size exceeds size of input image.')
    # pad kernel if necessary
    if np.any(np.greater(szF, szH)):
        if verbose:
            print('Kernel size is less than size of input image.')
            print('Zero-pad kernel to match it to the input image.')
        kernel = padpsf(kernel,szF)
    # check extension
    pad_width = tuple(map(lambda x:(x,x),minsupsz(kernel)))
    if verbose:
        print(pad_width)
    if extension[0:4] in ['circ', 'peri', 'symm']:
        if verbose:
            print('Extension type: ' + extension)
        extension = extension.lower()
        if extension[0:4] == 'circ':
            return np.real(ifftn(fftn(f)*psf2otf(kernel)))
        kernel = np.pad(kernel,pad_width,mode='constant',constant_values=0)
        f = np.pad(f,pad_width,mode=extension)
    else:
        if verbose:
            print('Number-like extension: ' + str(extension))
        if np.isscalar(extension):
            kernel = np.pad(kernel,pad_width,mode='constant',constant_values=0)
            f = np.pad(f,pad_width,mode='constant',constant_values=(extension,))
        else:
            raise ValueError('Only constant (scalar) number-like extension'\
                + ' is supported.')
    g = np.real(ifftn(fftn(f)*psf2otf(kernel)))
    for k in range(0,np.ndim(g)):
        g = np.delete(g,np.s_[0:pad_width[k][0]],k)
        g = np.delete(g,np.s_[szF[k]:],k)
    return g

def psf2otf(psf):
    """Discard the imaginary part of the otf if it's within round-off error.
    """
    otf = fftn(ifftshift(psf))
    npix = np.double(np.size(psf))
    szH = np.double(np.shape(psf))
    nops = np.sum(np.log2(szH)*npix)
    if np.max(np.abs(np.imag(otf)))/np.max(np.abs(otf)) <= nops*DEPS:
        return np.real(otf)
    else:
        return otf

def padpsf(kernel,pad_shape):
    """Pad PSF (convolution kernel function) to required shape.
    """
    szH = np.double(np.shape(kernel))
    szF = np.double(pad_shape)
    pad_width = tuple(map(lambda x,y: (0, np.int64(x-y)), szF, szH))
    kernel = np.pad(kernel,pad_width,mode='constant',constant_values=0)
    for k in range(0,np.ndim(kernel)):
        kernel = np.roll(kernel,\
            np.int(np.floor(szF[k]/2.0)-np.floor(szH[k]/2.0)),\
            axis=k)
    return kernel

def padpsf2d(kernel,pad_shape):
    """Pad PSF (convolution kernel function) to required shape.
    """
    szH = np.double(np.shape(kernel))
    szF = np.double(pad_shape)
    kernel = ifftshift(kernel)
    kernel = np.concatenate((kernel[0:np.int64(np.round(szH[0]*0.5)),:],\
        np.zeros([szF[0]-szH[0],szH[1]]),\
        kernel[np.int64(np.round(szH[0]*0.5)):szH[0],:]),axis=0)
    szH = np.shape(kernel)
    kernel = np.concatenate((kernel[:,0:np.int64(np.round(szH[1]*0.5))],\
        np.zeros([szH[0],szF[1]-szH[1]]),\
        kernel[:,np.int64(np.round(szH[1]*0.5)):szH[1]]),axis=1)
    kernel = fftshift(kernel)
    return kernel

def lastpow2(X):
    """Nearest integer power of 2 that is not greater than A.
    """
    return 2.0**np.floor(np.log2(X))

def nextpow2(X):
    """Nearest integer power of 2 that is not less than A.
    """
    return 2.0**np.ceil(np.log2(X))

def minsupsz(A,threshold=DEPS):
    """Minimum support size of input ndarray.
    """
    idx = np.nonzero(np.abs(A)>threshold)
    if np.size(idx) == 0:
        return (0,)*A.ndim
    else:
        return tuple(map(lambda x: np.max(x)-np.min(x)+1, idx))

def fdf(derivative=1, accuracy=2, mode='central'):
    """1D finite difference filter.
    """
    mode = mode.lower()
    if mode[0] == 'c':
        return cfdf(derivative,accuracy)
    elif mode[0] == 'f':
        return bfdf(derivative,accuracy)
    elif mode[0] == 'b':
        return ffdf(derivative,accuracy)
    else:
        raise StandardError('Unsupported mode '+mode)

def cfdf(derivative=1, accuracy=2):
    """1D central finite difference filter.
    """
    a = max(2, np.int(np.ceil(accuracy*0.5)*2.0))
    M = max(0, np.int(derivative))
    N = a+M-1
    alpha = reduce(lambda x,y:x+[-y,y],range(1,np.int(np.ceil(N/2.0)+1)),[0])
    c,alpha = fdiffcoef(M,N,0.0,alpha)
    f = np.zeros(len(alpha))
    f[alpha-np.min(alpha)] = np.double(c[M,N,:])[:]
    return f
    
def bfdf(derivative=1,accuracy=1):
    """1D backward finite difference filter.
    """
    a = max(1, accuracy)
    M = max(0, derivative)
    N = a+M-1
    alpha = range(0,-N-1,-1)
    c,alpha = fdiffcoef(M,N,0.0,alpha)
    return np.concatenate((np.double(c[M,N,:])[::-1],np.zeros(N)))

def ffdf(derivative,accuracy):
    """1D forward finite difference filter.
    """
    a = max(1, accuracy)
    M = max(0, derivative)
    N = a+M-1
    alpha = range(0,N+1)
    c,alpha = fdiffcoef(M,N,0.0,alpha)
    return np.concatenate((np.zeros(N),c[M,N,:]))

def fdiffcoef(M=4,N=8,x0=0.0,alpha=[0.0,-1.0,1.0,-2.0,2.0,-3.0,3.0,-4.0,4.0]):
    """
    Finite difference coefficients approximates M-order derivative with N+1
    grid points alpha={alpha_0, alpha_1, ..., alpha_N} around x0.
    Reference:
    Bengt Fornberg, Generation of finite difference formulas on arbitrarily
    spaced grids. Mathematics of Computation, Vol. 51, No. 184, 1988.
    """
    d = np.zeros([M+1,N+1,N+1])
    d[0,0,0] = 1.0
    c1 = 1.0
    for n in range (1,N+1):
        c2 = 1.0
        for v in range(0,n):
            c3 = alpha[n] - alpha[v]
            c2 = c2*c3
            # if n <= M:
                # d[n,n-1,v] = 0
            for m in range(0,min(n,M)+1):
                if m == 0:
                    d[m,n,v] = (alpha[n]-x0)*d[m,n-1,v]/c3
                else:
                    d[m,n,v] = (((alpha[n]-x0)*d[m,n-1,v])-m*d[m-1,n-1,v])/c3
        for m in range(0,min(n,M)+1):
            if m == 0:
                d[m,n,n] = -c1/c2*(alpha[n-1]-x0)*d[m,n-1,n-1]
            else:
                d[m,n,n] = c1/c2*(m*d[m-1,n-1,n-1]-(alpha[n-1]-x0)*d[m,n-1,n-1])
        c1 = c2
    return d,alpha[0:(N+1)]

def parse_inputs(dict_in,dict_out):
    """Parse variable-length keyworded input arguments.
    Parameters:
    dict_in  is input dictionary.
    dict_out is output dictionary.
    Values whose keys exist in dict_out will be pop out from dict_in
    to dict_out.
    """
    for k in dict_out:
        if dict_in.has_key(k):
            data_in = dict_in.pop(k)
            if data_in is not None:
                dict_out[k] = data_in
    return dict_in,dict_out

## define OpenCL functions.
try:
    import pyopencl as cl
    import pymath.clfuns as clfuns
    preferred_cl_device = clfuns.select_compute_device()
    def cl_rgb_to_hsl(r, g, b, device=None, profiling=False):
        """OpenCL implemented RGB to HSL color system transform.
"""
        ## prepare CL device & kernel source
        global preferred_cl_device
        if device is None:
            device = preferred_cl_device
        else:
            device = clfuns.find_compute_device(device)
        assert isinstance(device, cl.Device)
        ## check input image pixel format
        assert r.dtype in (
            np.dtype('uint8'), np.dtype('uint16'), np.dtype('uint32'), np.dtype('uint64'),
            np.dtype('float16'), np.dtype('float32'), np.dtype('float64')
        )
        if r.dtype.kind == 'u':
            is_uint = 1
        elif r.dtype.kind == 'f':
            is_uint = 0
        bits = 8 * r.dtype.itemsize
        npts = r.size
        with open(path.join(path.split(path.normpath(path.abspath(path.realpath(__file__))))[0], 'colorsys.c'), 'r') as fp:
            devsrc = fp.read()
        ## 
        if device.type == cl.device_type.GPU:
            ## GPU strategy:
            ## Many wavefronts per compute unit in order to encourage the native scheduler of each compute unit.
            print(u'{:<40}: {:<60}'.format('Device Type', 'GPU'))
            ws = 64 ## work size per compute unit
            global_work_size = compute_units * magic_number * ws
            if global_work_size > npts:
                global_work_size = npts
            else:
                while (global_work_size < npts) and ((npts % global_work_size) != 0):
                    global_work_size += ws
            local_work_size = ws
            is_cpu = 0
        else:
            ## CPU strategy:
            ## one thread per core.
            print(u'{:<40}: {:<60}'.format('Device Type', 'CPU'))
            global_work_size = compute_units
            local_work_size = 1
            is_cpu = 1
        ctx = cl.Context([device])
        if profiling:
            queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        r_cl = cl.Buffer(ctx, mf.READ_ONLY,  size=r.nbytes)
        g_cl = cl.Buffer(ctx, mf.READ_ONLY,  size=g.nbytes)
        b_cl = cl.Buffer(ctx, mf.READ_ONLY,  size=b.nbytes)
        h_cl = cl.Buffer(ctx, mf.WRITE_ONLY, size=r.nbytes)
        s_cl = cl.Buffer(ctx, mf.WRITE_ONLY, size=r.nbytes)
        l_cl = cl.Buffer(ctx, mf.WRITE_ONLY, size=r.nbytes)
        prg  = cl.Program(ctx, devsrc).build('-DIS_CPU={:d} -DIS_UINT={:d} -DBITS={:d} -DNITEMS={:d} -DCOUNT={:d}'.format(
            is_cpu, is_uint, bits, npts, int(npts/global_work_size)))
        preferred_multiple = cl.Kernel(prg, 'memcopy').get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            device
        )
        print(u'{:<40}: {:d}'.format('Data Size', int(npts)))
        print(u'{:<40}: {:d}'.format('Global Work Size', int(global_work_size)))
        print(u'{:<40}: {:d}'.format('Local Work Size', int(local_work_size)))
        print(u'{:<40}: {:d}'.format('Preferred Work Group Size Multiple', int(preferred_multiple)))
        i = 0
        p = 1.0
        t_push = []
        t_pull = []
        t_exec = []
        t_push_avg = 0.
        t_pull_avg = 0.
        t_exec_avg = 0.
        print(u'{:=^80}'.format(' Benchmark Records '))
        print(u'{:-<5}+{:-<20}+{:-<20}+{:-<20}'.format('', '', '', ''))
        print(u'{:<5}|{:<20}|{:<20}|{:<20}'.format('', ' t_push', ' t_exec', ' t_pull'))
        print(u'{:-<5}+{:-<20}+{:-<20}+{:-<20}'.format('', '', '', ''))
        while (i < max_repeats) and (p > precision):
            ev_push = cl.enqueue_copy(queue, a_cl, a_np)
            ev_exec = prg.memcopy(queue, (global_work_size,), (local_work_size,), a_cl, b_cl)
            ev_pull = cl.enqueue_copy(queue, b_np, b_cl)
            ##assert np.allclose(b_np, a_np)
            t_push.append(1e-9 * (ev_push.profile.end - ev_push.profile.start))
            t_exec.append(1e-9 * (ev_exec.profile.end - ev_exec.profile.start))
            t_pull.append(1e-9 * (ev_pull.profile.end - ev_pull.profile.start))
            p_push = np.abs(np.mean(t_push)-t_push_avg) / np.mean(t_push)
            p_exec = np.abs(np.mean(t_exec)-t_exec_avg) / np.mean(t_exec)
            p_pull = np.abs(np.mean(t_pull)-t_pull_avg) / np.mean(t_pull)
            t_push_avg = np.mean(t_push)
            t_exec_avg = np.mean(t_exec)
            t_pull_avg = np.mean(t_pull)
            p = max(p_push, p_exec, p_pull)
            print(u' {:>3d} | {:<8.2E} \u00b1 {:<7.1E} | {:<8.2E} \u00b1 {:<7.1E} | {:<8.2E} \u00b1 {:<7.1E}'.format(
                i, t_push_avg, np.std(t_push), t_exec_avg, np.std(t_exec), t_pull_avg, np.std(t_pull)))
            i += 1
        print(u'{:-<5}+{:-<20}+{:-<20}+{:-<20}'.format('', '', '', ''))
        print(u'{:=^80}'.format(' Benchmark Results '))
        r_host2dev = 8*npts / (np.double(t_pull) + np.double(t_push))
        r_global   = 8*npts / np.double(t_exec)
        return np.mean(r_host2dev), np.mean(r_global)
            
except:
    print("OpenCL functions are not available.")

def test():
    print('test bilinear...')
    f = misc.ascent()
    xi = np.arange(0,f.shape[1]-1,0.5)
    yi = np.arange(0,f.shape[0]-1,0.5)
    xi,yi = np.meshgrid(xi,yi)
    fi = bilinear([],[],f,xi,yi)
    plt.ion()
    plt.imshow(fi)
    print('test imconv...')
    hmotion = np.array([np.ones(50)])
    fmotion = imconv(f,hmotion,'symmetric')
    plt.figure(),plt.imshow(fmotion)
    vmotion = np.array([np.ones(50)]).T
    fmotion = imconv(f,vmotion,'symmetric')
    plt.figure(),plt.imshow(fmotion)
    print('test imtrans...')
    x = np.arange(0,512)-256.0
    y = np.arange(0,512)-256.0
    u = np.arange(0,725.0)-362.0
    v = np.arange(0,725.0)-362.0
    imrotatefunc = lambda theta:lambda x,y:(np.cos(theta)*x-np.sin(theta)*y,np.sin(theta)*x+np.cos(theta)*y)
    plt.figure(),plt.imshow(imtrans(x,y,f,imrotatefunc(np.pi/15.0),u,v))
