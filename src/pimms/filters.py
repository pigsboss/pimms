from pimms.base import *


def hdr_reinhard(I, m=None, f=0., c=0., a=0.):
    """HDR to LDR (dynamic range reduction) by Reinhard & Devlin 2005.
I - input image (HDR image), shape: (3, H, W), (H, W, 3) or (H, W)
m - contrast parameter, [0.3, 1.0]
f - intensity parameter, [-8, 8]
c - chromatic adaptation parameter, [0, 1]
a - light adaptation parameter, [0, 1]
"""
    ff = np.exp(-f)
    if m is None:
        Lavg = np.log(np.mean(I))
        Lmax = np.log(np.max(I))
        Lmin = np.log(np.min(I))
        k    = (Lmax - Lavg)/(Lmax - Lmin)
        m    = .3 + .7*(k**1.4)
    if np.ndim(I) == 3:
        lum = .2126*I[:,:,0] + .7152*I[:,:,1] + .0722*I[:,:,2]  # (Rec. 709, HDTV)
        lav = np.mean(lum)
        l = np.empty_like(I)
        g = np.empty_like(I)
        for i in range(3):
            cav = np.mean(I[:,:,i])
            l[:,:,i] = c*I[:,:,i] + (1.-c)*lum
            g[:,:,i] = c*cav      + (1.-c)*lav
        O = a*l + (1.-a)*g
    else:
        O = a*I + (1.-a)*np.mean(I)
    return I/(I+(ff*O)**m)

def deconvwnr(g,h,nsr):
    """Wiener deconvolution.
"""
    G = fftn(g)
    H = psf2otf(padpsf(h,np.shape(g)))
    W = np.conj(H) / (np.abs(H)**2 + nsr)
    return np.real(ifftn(G*W))

def multiresupport(X,J=None,k=3,epsilon=1e-3):
    """Multiresolution support.
    Noise of X follows Gaussian distribution.
    Reference:
      Starck, JL, Astronomical image and data analysis, 2006.
"""
    C,W = dwt(X,J)
    _,_,J = np.shape(W)
    sigma = auto_noise_estimate(X,J,k,epsilon)
    print('Estimated noise standard deviation: {:f}'.format(sigma))
    T = k*sigma*np.reshape(MR_NORMAL_STD2[0:J],(1,1,J))
    return np.array(np.abs(W)>=T,dtype=bool)

def wavelet_threshold(X,J=None,k=3,epsilon=1e-3):
    """Non-iterative wavelet threshold denoising.
    Reference:
      Starck, JL, Astronomical image and data analysis, 2006.
"""
    C,W = dwt(X,J)
    _,_,J = np.shape(W)
    sigma = auto_noise_estimate(X,J,k,epsilon)
    print('Estimated noise standard deviation: {:f}'.format(sigma))
    T = k*sigma*np.reshape(MR_NORMAL_STD2[0:J],(1,1,J))
    M = np.array(np.abs(W)>=T,dtype=bool)
    X = C + np.sum(W*M,axis=2)
    return X

def iterative_wavelet_threshold(X,J=None,k=3,epsilon=1e-3):
    """Non-iterative wavelet threshold denoising.
    Reference:
      Starck, JL, Astronomical image and data analysis, 2006.
"""
    R = []
    S = np.zeros(np.shape(X))
    n = 0
    C,W = dwt(X,J)
    _,_,J = np.shape(W)
    sigma = auto_noise_estimate(X,J,k,epsilon)
    print('Estimated noise standard deviation: {:f}'.format(sigma))
    T = k*sigma*np.reshape(MR_NORMAL_STD2[0:J],(1,1,J))
    delta = 1
    s_old = 0
    s_new = 0
    M = np.array(np.abs(W)>=T,dtype=bool)
    while delta > epsilon:
        R.append(X - S)
        s_new = np.std(R[n])
        CR,WR = dwt(R[n],J)
        R[n] = np.sum(WR*M,axis=2)+CR
        delta = np.abs(s_old - s_new)/s_new
        s_old = s_new
        S = S+R[n]
        n += 1
    return S,R
