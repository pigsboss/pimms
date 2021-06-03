"""This module provides solvers and auxiliary functions to find solutions of
convolution equations and modulation equations.
Solvers:
RL - Richardson-Lucy iteration for shift-invariant convolution equation.
VRL - Richardson-Lucy iteration for shift-variant convolution equation.
JRL - Richardson-Lucy iteration for joint shift-invariant convolution
        equations.
clean
iclean

"""
from pimms.base import *
from scipy.fftpack import ifft2,fft2,ifft,fft
from scipy.misc import ascent
from scipy.stats import skew
from time import time
from pimms.detectors import sex
from pymath.kerana import km2psf
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import sys
import tables
import multiprocessing as mp

MATRIX_PROD_IN_MEMORY=True

def estimate_source_flux(d, h, d_n, eps=1e-3, max_numit=1000):
    a = 1
    r = 10
    l = 0
    w = np.sum(h)
    m = np.logical_or(h>0, d_n>0)
    while np.abs(r-1.0) > eps and l < max_numit:
        r = np.sum(d[m]*h[m]/(a*h[m]+d_n[m]))/w
        a = r*a
        l += 1
    return a

def __find_source_worker__(q, worker_id, nworkers, d, d_n=0.0, kernel_matrix=None, kernel_matrix_file=None):
    if kernel_matrix is None:
        h5file, h5obj = kernel_matrix_file.split(":")
        f = tables.open_file(h5file,'r')
        H = f.get_node(h5obj)
    else:
        H = kernel_matrix
    nrows, ncols = H.shape
    flux = np.zeros(ncols)
    likelihood = np.zeros(ncols)
    for i in range(worker_id, ncols, nworkers):
        h = H[:, i]
        m = np.logical_or(h>0, d_n>0)
        a = estimate_source_flux(d[m], h[m], d_n[m])
        flux[i] = a
        likelihood[i] = np.sum(d[m]*np.log(a*h[m]+d_n[m])-(a*h[m]+d_n[m]))
    if kernel_matrix is None:
        f.close()
    q.put((flux,likelihood))

def find_source(d, kernel_matrix=None, kernel_matrix_file=None, d_n=0.0):
    """Find point source location and amplitude with maximum likelihood.

Observation model:
d = Hf + d_n,
d is observed data, poisson noise dominant.
H is kernel matrix
f is unknown object consists a single point source.
d_n is non-aperture modulated background of d, e.g., dark current.

H is given in either ND-array (kernel matrix) or HDF5 array (kernel matrix file name).
"""
    ncpus = mp.cpu_count()
    workers = []
    q = mp.Queue(maxsize=ncpus)
    for i in range(ncpus):
        worker = mp.Process(target=__find_source_worker__,
                            args=(q, i, ncpus, d, d_n, kernel_matrix, kernel_matrix_file))
        worker.start()
        workers.append(worker)
    flux, likelihood = q.get()
    for i in range(ncpus-1):
        pack = q.get()
        flux += pack[0]
        likelihood += pack[1]
    for worker in workers:
        worker.join()
    return flux, likelihood

def clean(I, H, Msk=None, R=None, C=None, S_Rect=None, beta=0.1, max_loop=100, beam_sigma=0.5):
    """Original CLEAN algorithm.

I is the dirty map.
Msk is the mask of I. 0 indicates bad pixels while 1 indicates good ones.
H is the dirty beam (PSF).
R is the residual map.
C is the clean map.
S_Rect is the rectangle area of possible sources, in [imin, imax, jmin, jmax], where i is row index and j in column index.
beta is the gain factor of the CLEAN algorithm.
beam_sigma is value of sigma of the ideal gauss-shaped beam in pixels.
max_loop is the maximum number of the CLEAN loops.
    """
    if Msk is None:
        Msk = np.ones(I.shape, dtype=I.dtype)
    if R is None:
        R = np.copy(I)
    if C is None:
        C = np.zeros(np.shape(I), dtype=I.dtype)
    if max_loop is None:
        print 'No maximum number of loops specified. Use 100 by default.'
        max_loop = 100
    if S_Rect is None:
        imin = 0
        imax = I.shape[0]
        jmin = 0
        jmax = I.shape[1]
    else:
        imin, imax, jmin, jmax = S_Rect
    N, M = np.shape(H)
    H_is_PSF = False
    if N == R.shape[0] and M == R.shape[1]:
        print 'Shift-invariant PSF input.'
        xc = np.argmax(np.max(H, axis=0))
        yc = np.argmax(np.max(H, axis=1))
        H_is_PSF = True
    clean_beam = gauss(R.shape, sigma=beam_sigma)
    x0 = np.argmax(np.max(clean_beam, axis=0))
    y0 = np.argmax(np.max(clean_beam, axis=1))
    l = 0
    while (np.max(R) > 0.0) and (np.max(R) > np.mean(R)) and \
        (skew(R.ravel()) > 0.0) and (l < max_loop):
        # locate maximum value
        xm = np.argmax(np.max(R[imin:imax,jmin:jmax], axis=0))+jmin
        ym = np.argmax(np.max(R[imin:imax,jmin:jmax], axis=1))+imin
        # get PSF
        if H_is_PSF:
            dirty_beam = subshift(H,(ym-yc,xm-xc))
        else:
            dirty_beam = km2psf(H, (ym,xm))
        # calculate optimal beta
        optbeta = getmtvamp(R, np.max(R)*dirty_beam)
        print optbeta
        if optbeta<0:
            break
        # update residual map and clean map
        C += beta*optbeta*np.max(R)*subshift(clean_beam,(ym-y0,xm-x0))
        R -= beta*optbeta*np.max(R)*dirty_beam*Msk
        l += 1
    return C,R

def totalvar(I):
    """Calculate total variation of given image.
    """
    return np.sqrt(np.sum((I - np.roll(I,1,axis=0))**2.0+\
        (I - np.roll(I,1,axis=1))**2.0))

def getmtvamp(I,P):
    """Get total-variation-minimizing amplitude of a component with the given
    image.
    I is the input image.
    P is the template of the component.
    R = I - A x P, where R is the residual image, which is supposed to be
    smooth, i.e., with minimized total variation.
    """
    ILx = I - np.roll(I,1,axis=1)
    ILy = I - np.roll(I,1,axis=0)
    PLx = P - np.roll(P,1,axis=1)
    PLy = P - np.roll(P,1,axis=0)
    return np.sum(ILx*PLx+ILy*PLy) / np.sum(PLx**2.0+PLy**2.0)

def iclean(I,H,R=None,C=None,imfilter=None,beta=0.2,max_loop=10,beam_sigma=0.5):
    """Improved CLEAN algorithm.
    1. Local maxima are detected by SExtractor.
    2. Image is filtered by specified function before detection.
    I is input image.

    H is the kernel matrix.

    imfilter is a callable object used as I_out = imfilter(I), where im_out
    is the filtered image which is used for source detection. If none imfilter
    is specified the original image I will be used for source detection
    directly.

    beta is the gain factor used in CLEAN algorithm.

    max_loop is upper limit of number of CLEAN loops.
    """
    if max_loop is None:
        print 'No maximum number of loops specified. Use 100 by default.'
        max_loop = 100
    if R is None:
        R = np.copy(I)
    if C is None:
        C = np.zeros(np.shape(I), dtype=I.dtype)
    l = 0
    num_pts = 1
    clean_beam = gauss(R.shape,beam_sigma)
    xc = 0.5*(1.0+R.shape[1])
    yc = 0.5*(1.0+R.shape[0])
    N, M = np.shape(H)
    H_is_PSF = False
    if N == R.shape[0] and M == R.shape[1]:
        print 'Shift-invariant PSF input.'
        H_is_PSF = True
    if imfilter is None:
        imfilter = lambda x:x
    watchlist = []
    delta = 1.0
    R_old = np.copy(R)
    while (num_pts > 0) and \
        (np.max(R) > 0.0) and \
        (np.max(R) > np.mean(R)) and \
        (skew(R.ravel()) > 0.0) and \
        (l < max_loop) and \
        (delta > 1e-6):
        F = imfilter(R)
        srclist = sex(F)[0]
        ptslist = filter(lambda s:s['FLAGS']<8,srclist)
        num_pts = len(ptslist)
        print 'CLEAN %d: %d point sources detected.'%(l,num_pts)
        for pts in ptslist:
            print '%f at (%f, %f)'%(pts['FLUX_ISO'],pts['X_IMAGE'],pts['Y_IMAGE'])
            if H_is_PSF:
                dirty_beam = subshift(H,(pts['Y_IMAGE']-yc,pts['X_IMAGE']-xc))
            else:
                j = np.uint64(np.round(pts['Y_IMAGE']-1.0)*R.shape[1]+np.round(pts['X_IMAGE']-1.0))
                try:
                    dirty_beam = subshift(np.reshape(H[:,j],R.shape),\
                        (pts['Y_IMAGE']-1.0-np.round(pts['Y_IMAGE']-1.0),\
                        pts['X_IMAGE']-1.0-np.round(pts['X_IMAGE']-1.0)))
                except:
                    print j
                    print R.shape
                    return H[:,j]
            optbeta = getmtvamp(R, pts['FLUX_ISO']*dirty_beam)
            print 'Beta: %f, optimal beta: %f'%(beta, optbeta)
            if optbeta<0:
                print 'Ignore negative optimal beta.'
                continue
            else:
                C += optbeta * pts['FLUX_ISO'] * subshift(clean_beam,\
                    (pts['Y_IMAGE']-yc,pts['X_IMAGE']-xc))
                R -= optbeta * pts['FLUX_ISO'] * dirty_beam

        delta = np.std(R - R_old)
        R_old = np.copy(R)
        l += 1
        watchlist.append((C,R,F,totalvar(R)))
    return C,R,F,watchlist

def factorial(n):
    """Compute factorial of integer scalar n.
    """
    return np.prod(np.arange(1,n+1))

def vectorextrap(f,g,T,accelerate):
    """Predict an estimate by vector extrapolation.
    f is array of previous estimates
    g is array of previous gradients
    T is taylor coefficients matrix
    accelerate is order of Acceleration
    y is predicted estimate for output
    """
    # calculate step size
    if accelerate == 0:
        return f[0]
    g01 = np.sum(g[0]*g[1])
    g11 = np.sum(g[1]**2.0)
    if g01 <= np.finfo(g.dtype).eps:
        return f[0]
    gamma  = np.power(np.clip(g01/(g11+np.finfo(g.dtype).eps),0.0,1.0), 1.0/accelerate)
    gamman = gamma ** np.arange(0.0, accelerate+1.0)
    TG = np.matmul(T, gamman.T)
    y  = np.sum(TG.reshape((accelerate+1,)+(1,)*f[0].ndim)*f[0:(accelerate+1)], axis=0)
    return y

def taylorcoefm(n):
    """Taylor coefficients matrix of n-th order.
    """
    T = np.zeros((n+1,n+1),dtype='float64')
    for k in range(n+1):
        T[k,k:(n+1)] = (1.0/np.array(map(factorial,range(n-k+1)),\
            dtype='float64'))*(1.0/factorial(k))*(-1.0)**np.double(k)
    return T

def make_mask_matrix(v,M):
    """A mask matrix is a diagonal matrix.
    v is the diagonal of the matrix.
    M is the output matrix.
    """
    for i in range(np.size(v)):
        M[i,i] = v[i]

def matrix_prod(A,B,C):
    """C = A x B
    """
    if MATRIX_PROD_IN_MEMORY:
        np.matmul(A,B,C)
    else:
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                C[i,j] = np.sum(A[i,:]*B[:,j])

def RL(d,H,f=None,g=None,lower_limit=0.0,d_n=0.0,
       upper_limit=None,regulator=None,accelerate=1,numit=20,show_progress=True):
    """Accelerated Richardson-Lucy iteration to solve shift-invariant convolution equations.

Modulation equation:
d = H * f + d_n,
where d is the observed data, H is the shift-invariant convolution kernel, i.e.,
the PSF. The PSF must not be larger than the observed data along any axis.
f is 3-d array of previous estimate(s) of the object image.
d_n is non-convolved-observed data, e.g., detector dark current, etc.

g is 3-d array of gradients of f used for acceleration of the iteration.

lower_limit and upper_limit are used as constraints. User input g must
be either returned from a previous run of this function or initialized
with all elements set to 0.

regulator is additional regularization function.

accelerate is 0, 1, 2, or 3. 0 stands for no acceleration. 1, 2 and 3
stand for linear, 2nd-order and 3rd-order accelerations.
Acceleration order higher than 3 is strongly discouraged.
Acceleration order higher than 9 is clipped to 9.

numit is the number of demanded iterations.
"""
    tic = time()
    # examine and normalize user inputs
    d,H,_,f,g,accelerate,K,N,M,numest = check_inputs(d,H,None,f,g,accelerate)
    # start accelerated fixed-point iteration
    T = taylorcoefm(accelerate)
    OTF = psf2otf(H)
    IOTF = np.conj(OTF)
    omega = np.ones((N,M),dtype='float64')*np.sum(H)
    if np.any(np.abs(f[0,:,:])<=np.finfo(f.dtype).eps):
        print 'Calculating initial estimate...'
        f[0,:,:] = d / omega
    print 'Preparation finished.'
    print 'RL iteration begins. %0.2f seconds elapsed.'%(time()-tic)
    if callable(regulator):
        print 'Additional regularization provided.'
        regularize = lambda x: regulator(np.clip(x, lower_limit, upper_limit))
    else:
        regularize = lambda x: np.clip(x, lower_limit, upper_limit)
    for l in range(numit):
        y = vectorextrap(f,g,T,accelerate)
        y = regularize(y)
        f = np.roll(f,1,axis=0)
        f[0,:,:] = regularize(np.real(ifft2(IOTF*fft2(d/np.clip(d_n + np.real(ifft2(OTF*fft2(y))),DEPS,None))))*y/omega)
        g[1,:,:] = g[0,:,:]
        g[0,:,:] = f[0,:,:] - y
        if show_progress and np.mod(l+1,np.ceil(numit/10.0)) == 0:
            print '%d steps (%0.1f%%) finished, %0.2f seconds elapsed.'%\
                (l+1,(l+1.0)*100.0/numit,time()-tic)
    return f,g

def MRL(d,H,HT=None,f=None,g=None,lower_limit=0.0,d_n=0.0,
        upper_limit=None,regulator=None,accelerate=1,numit=20,show_progress=True):
    """Accelerated Richardson-Lucy iteration to solve modulation equations.

Modulation equation:
d = Hf + d_n,
where d is the observed data. H is the modulation kernel matrix.
f is the unknown object. d_n is non-modulated observed data, e.g., dark current, etc.

HT is the transpose of H.

f is 3-d array of previous estimate(s) of the object image.

g is 3-d array of gradients of f used for acceleration of the iteration.

lower_limit and upper_limit are used as constraints. User input g must
be either returned from a previous run of this function or initialized
with all elements set to 0.

regulator is additional regularization function.

accelerate is 0, 1, 2, or 3. 0 stands for no acceleration. 1, 2 and 3
stand for linear, 2nd-order and 3rd-order accelerations.
Acceleration order higher than 3 is strongly discouraged.
Acceleration order higher than 9 is clipped to 9.

numit is the number of demanded iterations.
"""
    tic = time()
    # examine and normalize user inputs
    d = np.reshape(d, (-1,1))
    N,_ = np.shape(d)
    print 'Observed data: %d pixels, '%(N)+str(d.dtype)+'.'
    num_rows_H, M = np.shape(H)
    print 'Kernel matrix: %d x %d, '%(num_rows_H,M)\
        +str(H.dtype)+'.'
    if num_rows_H != N:
        raise StandardError('The shapes of the kernel matrix is wrong.')
    if accelerate > 3:
        print 'Acceleration order higher than 3 is strongly discouraged.'
        print 'Acceleration order higher than 9 is clipped to 9.'
    accelerate = np.uint8(np.clip(accelerate,0,9))
    numest = accelerate+1
    print 'Acceleration order: %d'%accelerate
    print '%d previous estimates needed.'%numest
    if accelerate == 0:
        print 'No previous gradients needed.'
    else:
        print '2 previous gradients needed.'
    if HT is None:
        HT = np.transpose(H)
    omega = np.reshape(np.sum(H,axis=0),(M,1))
    if f is None:
        print 'No previous estimates input.'
        f = np.zeros((numest,M,1),dtype='float64')
        matrix_prod(HT,d,f[0,:,:])
        f[0,:,:] = f[0,:,:] / (omega**2.0)
    if g is None:
        print 'No previous gradients input.'
        g = np.zeros((2,M,1),dtype=f.dtype)
    if callable(regulator):
        print 'Additional regularization provided.'
        regularize = lambda x: regulator(np.clip(x, lower_limit, upper_limit))
    else:
        regularize = lambda x: np.clip(x, lower_limit, upper_limit)
    # start accelerated fixed-point iteration
    T = taylorcoefm(accelerate)
    print 'Preparation finished.'
    print 'MRL iteration begins. %0.2f seconds elapsed.'%(time()-tic)
    for l in range(numit):
        y = vectorextrap(f,g,T,accelerate)
        y = regularize(y)
        f = np.roll(f,1,axis=0)
        r = np.empty(d.shape,dtype='float64')
        matrix_prod(H,y,r)
        r = d/np.clip(r+d_n, DEPS, None)
        matrix_prod(HT,r,f[0,:,:])
        f[0,:,:] = regularize(f[0,:,:]*y/omega)
        g[1,:,:] = g[0,:,:]
        g[0,:,:] = f[0,:,:] - y
        if show_progress and np.mod(l+1,np.ceil(numit/10.0)) == 0:
            sys.stdout.write('\r%d steps (%0.1f%%) finished, %0.1f seconds elapsed, %0.1f seconds left.'%\
                (l+1,(l+1.0)*100.0/numit,time()-tic,(time()-tic)*(numit-l-1.0)/(l+1.0)))
            sys.stdout.flush()
    print '\nComplete.'
    return f,g

def JRL(d,H,f=None,g=None,lower_limit=0.0,d_n=0.0,
        upper_limit=None,regulator=None,accelerate=1,numit=20,show_progress=True):
    """Accelerated Richardson-Lucy iteration to solve joint shift-invariant convolution equations.

Modulation equation:
d = H*f + d_n,
where d is the observed data. H is the shift-invariant convolution kernel, i.e., 
the PSF. The PSF must not be larger than the observed data along any axis.
d_n is non-modulated observed data, e.g., the dark current, etc.

f is 3-d array of previous estimate(s) of the object image.

g is 3-d array of gradients of f used for acceleration of the iteration.

lower_limit and upper_limit are used as constraints. User input g must
be either returned from a previous run of this function or initialized
with all elements set to 0.

regulator is additional regularization function.

accelerate is 0, 1, 2, or 3. 0 stands for no acceleration. 1, 2 and 3
stand for linear, 2nd-order and 3rd-order accelerations.
Acceleration order higher than 3 is strongly discouraged.
Acceleration order higher than 9 is clipped to 9.

numit is the number of demanded iterations.
"""
    tic = time()
    # examine and normalize user inputs
    d,H,_,f,g,accelerate,L,_,N,M,numest = check_joint_inputs(d,H,None,f,g,accelerate)
    # start accelerated fixed-point iteration
    T = taylorcoefm(accelerate)
    OTF = np.empty((L,N,M),'complex128')
    IOTF = np.empty((L,N,M),'complex128')
    for k in range(L):
        OTF[k,:,:] = psf2otf(H[k])
        IOTF[k,:,:] = np.conj(OTF[k,:,:])
    omega = np.ones((N,M),dtype='float64')*np.sum(H)
    if np.any(np.abs(f[0,:,:])<=np.finfo(f.dtype).eps):
        print 'Calculating initial estimate...'
        for l in range(L):
            f[0,:,:] += d[l]
        f[0,:,:] = f[0,:,:] / np.double(L) / omega
    print 'Preparation finished.'
    print 'JRL iteration begins. %0.2f seconds elapsed.'%(time()-tic)
    if callable(regulator):
        print 'Additional regularization provided.'
        regularize = lambda x: regulator(np.clip(x, lower_limit, upper_limit))
    else:
        regularize = lambda x: np.clip(x, lower_limit, upper_limit)
    for l in range(numit):
        y = vectorextrap(f,g,T,accelerate)
        y = regularize(y)
        f = np.roll(f,1,axis=0)
        f[0,:,:] = regularize(np.sum(np.array(np.real(ifft2(IOTF*fft2(d/\
            np.clip(d_n + np.real(ifft2(OTF*fft2(y))),\
            DEPS,None)))),ndmin=3),axis=0)*y/omega)
        g[1,:,:] = g[0,:,:]
        g[0,:,:] = f[0,:,:] - y
        if show_progress and np.mod(l+1,np.ceil(numit/10.0)) == 0:
            print '%d steps (%0.1f%%) finished, %0.2f seconds elapsed.'%\
                (l+1,(l+1.0)*100.0/numit,time()-tic)
    return f,g

def __VRL_reblur__(worker_id, nworkers, K, N, q_in, q_out, A, IOTF):
    S = q_in.get()
    while S is not None:
        a = np.zeros(N)
        for k in range(worker_id, K, nworkers):
            a += A[k].dot(np.real(ifft(S*IOTF[k])))
        q_out.put(a)
        S = q_in.get()

def __VRL_deblur__(worker_id, nworkers, K, M, q_in, q_out, AT, OTF, d, d_n):
    a = q_in.get()
    while a is not None:
        b = np.zeros(M)
        for k in range(worker_id, K, nworkers):
            b += np.real(ifft(fft(AT[k].dot(d/np.clip(a+d_n,DEPS,None)))*OTF[k]))
        q_out.put(b)
        a = q_in.get()

def VRL(d,H,A=None,f=None,g=None,lower_limit=0,d_n=0.0,
        upper_limit=None,regulator=None,accelerate=True,numit=20,show_progress=True):
    """Accelerated Richardson-Lucy iteration to solve shift-variant convolution equations.

Modulation equation:
d = sum(A Hf) + d_n,
where d is the array of observed data. 
H is the 2-d array of shift-invariant convolution kernels.
A is the 2-d array of coefficients of H.
d_n is non-modulated observed data, e.g., the dark current, etc.

f is 2-d array of previous estimate(s) of the object image.

g is 2-d array of gradients of f used for acceleration of the iteration.

lower_limit and upper_limit are used as constraints. User input g must
be either returned from a previous run of this function or initialized
with all elements set to 0.

regulator is additional regularization function.

accelerate is 0, 1, 2, or 3. 0 stands for no acceleration. 1, 2 and 3
stand for linear, 2nd-order and 3rd-order accelerations.
Acceleration order higher than 3 is strongly discouraged.
Acceleration order higher than 9 is clipped to 9.

numit is the number of demanded iterations.
"""
    # examine and normalize user inputs
    N   = d.size
    K,M = H.shape
    if A is None:
        if K==1 and M==N:
            A = np.array(np.diag(np.ones(M)), ndmin=3)
        else:
            raise StandardError("A (diagonal vector matrix) must be specified for more than one cluster.")
    if accelerate > 3:
        print 'Acceleration order higher than 3 is strongly discouraged.'
        print 'Acceleration order higher than 9 is clipped to 9.'
    acceelrate = np.uint8(np.clip(accelerate, 0, 9))
    numest     = accelerate + 1
    print 'Acceleration order: %d'%accelerate
    print '%d previous estimates needed.'%numest

    # prepare workspace
    tic   = time()
    T     = taylorcoefm(accelerate)
    OTF   = np.empty((K,M), dtype='complex128')
    IOTF  = np.empty((K,M), dtype='complex128')
    omega = np.zeros((M,),  dtype='float64')
    if sparse.isspmatrix(A[0]):
        AT = [A[k].transpose() for k in range(K)]
    else:
        AT = A.transpose((0,2,1))
    for k in range(K):
        otf = fft(H[k])
        if (np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf))) <= (np.log2(M)*M*DEPS):
            OTF[k] = np.real(otf)
        else:
            OTF[k] = otf
        IOTF[k] = np.conj(OTF[k])
        omega  += np.real(ifft(fft(AT[k].dot(np.ones(N))) * OTF[k]))
    if accelerate == 0:
        print 'No previous gradients needed.'
    else:
        print '2 previous gradients needed.'
    if f is None:
        print 'No previous estimates input.'
        print 'Calculating initial estimate...'
        f = np.zeros((numest, M), dtype = 'float64')
        for k in range(K):
            f[0] += np.real(ifft(fft(AT[k].dot(d)) * OTF[k]))
        f[0] = f[0] / (omega**2.0)
    if g is None:
        print 'No previous gradients input.'
        g = np.zeros((2, M), dtype='float64')

    # start accelerated fixed-point iteration
    print 'Preparation finished.'
    print 'VRL iteration begins. %0.2f seconds elapsed.'%(time()-tic)
    if callable(regulator):
        print 'Additional regularization provided.'
        regularize = lambda x: regulator(np.clip(x, lower_limit, upper_limit))
    else:
        regularize = lambda x: np.clip(x, lower_limit, upper_limit)
    nws = max(2,min(mp.cpu_count(),K))
    qri = mp.Queue()
    qro = mp.Queue()
    qdi = mp.Queue()
    qdo = mp.Queue()
    prs = []
    pds = []
    for i in range(nws):
        p = mp.Process(target=__VRL_reblur__, args=(i,nws,K,N,qri,qro,A,IOTF))
        p.start()
        prs.append(p)
    for i in range(nws):
        p = mp.Process(target=__VRL_deblur__, args=(i,nws,K,M,qdi,qdo,AT,OTF,d,d_n))
        p.start()
        pds.append(p)
    for l in range(numit):
        y = vectorextrap(f,g,T,accelerate)
        y = regularize(y)
        f = np.roll(f,1,axis=0)
        a = np.zeros(N)
        b = np.zeros(M)
        S = fft(y)
        for i in range(nws):
            qri.put(S)
        for i in range(nws):
            a += qro.get()
        for i in range(nws):
            qdi.put(a)
        for i in range(nws):
            b += qdo.get()
        f[0] = regularize(b*y/omega)
        g[1] = g[0]
        g[0] = f[0] - y
        if show_progress and np.mod(l+1,np.ceil(numit/10.0)) == 0:
            print '%d steps (%0.1f%%) finished, %0.2f seconds elapsed.'%\
                (l+1,(l+1.0)*100.0/numit,time()-tic)
    for i in range(nws):
        qri.put(None)
        qdi.put(None)
    for p in prs:
        p.join()
    for p in pds:
        p.join()
    return f,g

def check_inputs(d,H,A,f,g,accelerate):
    """Examine and normalize user inputs for SIRL and SVRL.
    """
    d = np.array(d,ndmin=2)
    N, M = np.shape(d)
    print 'Observed data: %d x %d, '%(N,M)+str(d.dtype)+'.'
    H = np.array(H,ndmin=3)
    K, num_rows_H, num_cols_H = np.shape(H)
    print 'PSF: %d x %d x %d, '%(K,num_rows_H,num_cols_H)\
        +str(H.dtype)+'.'
    if (num_cols_H != M) or (num_rows_H != N):
        if (num_cols_H > M) or (num_rows_H > N):
            raise StandardError('The PSF can not be larger'\
                ' than the observed data along any axis.')
        else:
            H_old = np.copy(H)
            H = np.empty((K,N,M),dtype='float64')
            for k in range(K):
                H[k,:,:] = padpsf(H_old[k,:,:],(N,M))
            print 'PSF is padded to %d x %d.'%(N,M)
    if accelerate > 3:
        print 'Acceleration order higher than 3 is strongly discouraged.'
        print 'Acceleration order higher than 9 is clipped to 9.'
    accelerate = np.uint8(np.clip(accelerate,0,9))
    numest = accelerate+1
    print 'Acceleration order: %d'%accelerate
    print '%d previous estimates needed.'%numest
    if accelerate == 0:
        print 'No previous gradients needed.'
    else:
        print '2 previous gradients needed.'
    if A is None:
        if K == 1:
            A = np.ones((1,N,M),dtype='float64')
        else:
            raise StandardError('A must be provided in case K > 1.')
    else:
        A = np.array(A,ndmin=3)
        num_A,num_rows_A,num_cols_A = np.shape(A)
        if (num_A != K) or (num_rows_A != N) or (num_cols_A != M):
            raise StandardError('Shape of A is wrong.')
    if f is None:
        print 'No previous estimates input.'
        f = np.zeros((numest,N,M),dtype='float64')
    if g is None:
        print 'No previous gradients input.'
        g = np.zeros((2,N,M),dtype=f.dtype)
    return d,H,A,f,g,accelerate,K,N,M,numest

def check_joint_inputs(d,H,A,f,g,accelerate):
    """Examine and normalize user inputs for SIRL and SVRL.
    """
    L, N, M = np.shape(np.array(d,ndmin=3))
    d = list(d)
    print 'Observed data: %d x %d x %d, '%(L,N,M)+str(d[0].dtype)+'.'
    H = list(H)
    if len(H) != L:
        print 'Number of PSFs is wrong.'
    K = np.empty(L,dtype='uint64')
    for l in range(L):
        H[l] = np.array(H[l],ndmin=3)
        K[l], num_rows_H, num_cols_H = np.shape(H[l])
        print 'PSF: %d x %d x %d, '%(K[l],num_rows_H,num_cols_H)\
            +str(H[l].dtype)+'.'
        if (num_cols_H != M) or (num_rows_H != N):
            if (num_cols_H > M) or (num_rows_H > N):
                raise StandardError('The PSF can not be larger'\
                    ' than the observed data along any axis.')
            else:
                H_old = np.copy(H[l])
                H[l] = np.empty((K[l],N,M),dtype='float64')
                for k in range(K[l]):
                    H[l][k,:,:] = padpsf(H_old[k,:,:],(N,M))
                print 'PSF is padded to %d x %d.'%(N,M)
    if accelerate > 3:
        print 'Acceleration order higher than 3 is strongly discouraged.'
        print 'Acceleration order higher than 9 is clipped to 9.'
    accelerate = np.uint8(np.clip(accelerate,0,9))
    numest = accelerate+1
    print 'Acceleration order: %d'%accelerate
    print '%d previous estimates needed.'%numest
    if accelerate == 0:
        print 'No previous gradients needed.'
    else:
        print '2 previous gradients needed.'
    if A is None:
        if np.any(K == 1):
            A = [np.ones((1,N,M),dtype='float64')] * L
        else:
            raise StandardError('A must be provided in case K > 1.')
    else:
        if len(A) != L:
            raise StandardError('Number of A is wrong.')
        A = list(A)
        for l in range(L):
            A[l] = np.array(A[l],ndmin=3)
            num_A,num_rows_A,num_cols_A = np.shape(A[l])
            if (num_A != K[l]) or (num_rows_A != N) or (num_cols_A != M):
                raise StandardError('Shape of A is wrong.')
    if f is None:
        print 'No previous estimates input.'
        f = np.zeros((numest,N,M),dtype='float64')
    if g is None:
        print 'No previous gradients input.'
        g = np.zeros((2,N,M),dtype=f.dtype)
    return d,H,A,f,g,accelerate,L,K,N,M,numest

def test(sigma=5,numit=20,accelerate=3,K=5,L=5):
    f = np.double(ascent())
    h = gauss(np.shape(f),sigma = sigma)
    d = imconv(f,h)
    F,G = RL(d,h,numit=numit,accelerate=accelerate,upper_limit=255)
    plt.set_cmap('gray')
    fig,axs = plt.subplots(nrows=2,ncols=2)
    axs[0][0].imshow(f)
    axs[0][0].set_title('object image')
    axs[0][1].imshow(h)
    axs[0][1].set_title('PSF')
    axs[1][0].imshow(d)
    axs[1][0].set_title('observed data')
    axs[1][1].imshow(F[0])
    axs[1][1].set_title('reconstructed image')
    fig.suptitle('shift-invariant RL')
    fig.show()
    N,M = np.shape(f)
    h = np.empty((K,N,M),dtype='float64')
    H = np.empty((K,N*M),dtype='float64')
    mask = imconv(np.random.rand(N,M),gauss((N,M),10.0))
    mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask))
    m = np.zeros((N,M))
    a = np.empty((K,N,M),dtype='bool')
    for k in range(K):
        h[k] = imrotate(gauss((N,M),sigma = (1,2.0*sigma)),angle=np.pi*k/K,box='crop')[0]
        H[k] = ifftshift(h[k]).ravel()
        a[k] = np.logical_and(mask>=(k/np.double(K)),mask<((k+1.0)/np.double(K)))
        m[a[k,:,:]] = k
    print np.sum(a) / (N*M)
    for k in range(K):
        for l in range(K):
            print np.sum(a[k]*a[l])
    d = np.zeros((N,M), dtype='float64')
    for k in range(K):
        d += a[k]*imconv(f,h[k])
    A   = [sparse.dia_matrix((a[k].ravel(),0), (N*M,N*M)) for k in range(K)]
    F,G = VRL(d.ravel(),H,A,numit=numit,accelerate=accelerate,upper_limit=255)
    fig,axs = plt.subplots(nrows=2,ncols=3)
    axs[0][0].imshow(f)
    axs[0][0].set_title('object image')
    axs[0][1].imshow(np.sum(h,axis=0))
    axs[0][1].set_title('PSF overlay')
    axs[0][2].imshow(m)
    axs[0][2].set_title('PSF partition')
    axs[1][0].imshow(a[K/2,:,:])
    axs[1][0].set_title('PSF %d'%(K/2))
    axs[1][1].imshow(d)
    axs[1][1].set_title('observed data')
    axs[1][2].imshow(F[0].reshape((N,M)))
    axs[1][2].set_title('reconstructed image')
    fig.suptitle('shift-variant RL')
    fig.show()
    d = []
    H = []
    for l in range(L):
        H.append(imrotate(gauss((512,512),(10,2)),angle=np.pi/L*l,box='crop')[0])
        d.append(imconv(f,H[l]))
    J,_=JRL(d,H)
    F,_=RL(np.mean(d,axis=0),np.mean(H,axis=0))
    fig,axs = plt.subplots(nrows=1,ncols=2)
    axs[0].imshow(F[0].reshape((N,M)))
    axs[0].set_title('Overlay')
    axs[1].imshow(J[0])
    axs[1].set_title('Joint')
    fig.show()
