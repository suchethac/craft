import numpy as np
import scipy.fftpack as fft
from scipy import constants
from scipy import ndimage

# To speed up the Fast Fourier Trasnforms, one can use 
# FFTW implementation for python as shown below.

# try:
#     import pyfftw
#     import multiprocessing
#     pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
#     pyfftw.interfaces.cache.enable()
#     import pyfftw.interfaces.scipy_fftpack as fft
# except:
#     import scipy.fftpack as fft
    

##### Unit Conversions #####

def MHz2m2(f): 
    """
    Converts frequencies in [MHz] to lambda squared value in [m^2].
    
    Parameters
    ----------
    f  : float
        A value of frequency in [MHz]

    Returns
    -------
    ls : float
        A value of lambda squared in [m^2]
    """
    f *= 1000000 # MHz to Hz
    lambda_ = constants.speed_of_light / f
    ls = pow(lambda_, 2)
    return ls

def m22MHz(lambda_squared):
    """
    Converts lambda squared value in [m^2] to frequencies in [MHz].
    
    Parameters
    ----------
    ls  : float
        A value of lambda squared in [m^2]
        
    Returns
    -------
    f : float
        A value of frequency in [MHz]
    """
    lambda_ = pow(lambda_squared, 0.5)
    f = constants.speed_of_light / lambda_
    f /= 1000000 # MHz to Hz 
    return f

def freq_range2ls_range(f_min, f_max):
    """
    Input the observed frequency range in MHz and get the range of the 
    observed linear polarization. Note that f_min (f_max) is used to 
    caluculate ls_max (ls_min).
    
    Parameters
    ----------
    f_min : float
        Lower limit of frequency in [MHz] of the observed spectrum
    f_max : float
        Upper limit of frequency in [MHz] of the observed spectrum

    Returns
    -------
    ls_min : float
        Lower limit of lamdba squared in [m^2] of the observed spectrum
    ls_max : float
        Upper limit of lamdba squared in [m^2] of the observed spectrum
    """
    ls_min, ls_max = MHz2m2(f_max), MHz2m2(f_min)
    return ls_min, ls_max

##############################

##### Scale Calculations #####

def rmsf_fwhm(ls_min, ls_max):
    """
    Calculate the full-width-at-half_maximum (FWHM) of the rotation
    measure spread function (RMSF).
    
    Parameters
    ----------
    ls_min : float
        Lower limit of lamdba squared in the observed spectrum
    ls_max : float
        Upper limit of lamdba squared in the observed spectrum

    Returns
    -------
    rmsf_fwhm_ : float
        FWHM of RMSF in phi
    """
    rmsf_fwhm_ = 2*pow(3, 0.5)/(ls_max - ls_min)
    return rmsf_fwhm_

def min_reconstruction_scale(ls_min, ls_max):
    """
    Calculate the smallest scale in phi that can be reconstructed 
    successfully. The calculation is given by Eq. (9) in Cooray et al.
    2020b. 
    
    Parameters
    ----------
    ls_min : float
        Lower limit of lamdba squared in the observed spectrum
    ls_max : float
        Upper limit of lamdba squared in the observed spectrum
    
    Returns
    -------
    min_recon_scale : float
        Smallest reconstruction scale in phi
    """
    min_recon_scale = pow(3, 0.5)/ls_max
    return min_recon_scale

def chi_smoothing_scale(ls_min, ls_max):
    """
    Calculate the FWHM of the Gaussian kernel used to smooth the 
    polarization angle at each iteration. The smoothing scale in
    phi is set as the main lobe width of the RMSF given as;

    smoothing_scale = 2*\pi / (ls_max-ls_min) 
    
    Parameters
    ----------
    ls_min : float
        Lower limit of lamdba squared in the observed spectrum
    ls_max : float
        Upper limit of lamdba squared in the observed spectrum

    Returns
    -------
    smoothing_scale : float
        FWHM of the Gaussian kernel used polarization angle 
        smoothing
    """
    smoothing_scale = 2*np.pi/(ls_max-ls_min)
    return smoothing_scale
    
##############################

##### Fourier transforms #####

def FT_1D(ls, P, axis=-1):
    """
    Fourier transform the Faraday dispersion function F(phi) to 
    obtain the complex linear polarization spectrum P(lambda^2).
    The function uses the FFT to approximate the continuous Fourier 
    transform of a discretely sampled function.
    
       F(phi) = integral[ P(ls) exp(-2*i*phi*ls) dls]
       P(ls) = integral[ F(phi) exp(2*i*phi*ls) dphi]
       
    Function returns phi and F, which approximate F(phi).
    
    Parameters
    ----------
    ls  : array_like
        regularly sampled array of lambda_squared.
        ls is assumed to be regularly spaced, i.e.
        ls = ls0 + Dls * np.arange(N)
    P   : array_like
        Complex linear polarization spectrum.
    axis : int
        axis along which to perform fourier transform.

    Returns
    -------
    phi : ndarray
        Faraday depth of the calculated Faraday dispersion function.
    F   : ndarray
        Complex Faraday dispersion function.
    """
    assert ls.ndim == 1
    assert P.shape[axis] == ls.shape[0]
    
    N = len(ls)
    if N % 2 != 0:
        raise ValueError("Number of samples must be even")     
    
    shape = np.ones(P.ndim, dtype=int)
    shape[axis] = N
    
    Dls = ls[1] - ls[0]
    Dphi = 1. / (N * Dls)
    ls0 = ls[int(N / 2)]

    phi = Dphi * (np.arange(N) - N / 2)

    phase = np.ones(N)
    phase[1::2] = -1
    phase = phase.reshape(shape)

    F = Dls * fft.fft(P * phase, axis=axis)
    F *= phase
    F *= np.exp(-2j * np.pi * ls0 * phi.reshape(shape))
    F *= np.exp(-1j * np.pi * N / 2)
    
    phi*=np.pi
    
    return phi, F

def IFT_1D(phi, F, axis=-1):
    """
    Inverse Fourier transform the complex linear polarization spectrum 
    P(lambda^2) to obtain the Faraday dispersion function F(phi).
    The function uses the FFT to approximate the inverse continuous 
    Fourier transform of a discretely sampled function.
    
       F(phi) = integral[ P(ls) exp(-2*i*phi*ls) dls]
       P(ls) = integral[ F(phi) exp(2*i*phi*ls) dphi]
       
    Function returns ls and P, which approximate P(ls).
    
    Parameters
    ----------
    phi : array_like
        regularly sampled array of Faraday depth phi.
        phi is assumed to be regularly spaced, i.e.
        phi = phi0 + Dphi * np.arange(N)
    F   : array_like
        Complex Faraday dispersion function
    axis : int
        axis along which to perform fourier transform.

    Returns
    -------
    ls  : ndarray
        lambda squared of the calculated linear polarization 
        spectrum.
    P   : ndarray
        Complex linear polarization spectrum.
    """
    assert phi.ndim == 1
    assert F.shape[axis] == phi.shape[0]
    
    N = len(phi)
    if N % 2 != 0:
        raise ValueError("Number of samples must be even")
        
    shape = np.ones(F.ndim, dtype=int)
    shape[axis] = N
    
    phi = phi/np.pi
    phi0 = phi[0]
    Dphi = phi[1] - phi[0]
    ls0 = -1. / 2*Dphi
    Dls = 1. / (N * Dphi)
    ls = ls0 + Dls * np.arange(N)

    ls_calc = ls.reshape(shape)
    phi_calc = phi.reshape(shape)

    F_prime = F * np.exp(2j * np.pi * ls0 * phi_calc)
    P_prime = fft.ifft(F_prime, axis=axis)
    P = N * Dphi * np.exp(2j * np.pi * phi0 * (ls_calc - ls0)) * P_prime

    return ls, P

##############################

##### Reconstruction algorithm #####

def normalized_residual(current, previous):
    '''
    A helper function to calculate the relative difference of each iteration
    step. Calculated as the normalized norm or the residual,

    normalized_residual = ||f_(n+1) - f_n|| / ||f_n||

    The returned value is between 0 and 1.
    '''
    res = previous - current
    r = np.linalg.norm(res)/np.linalg.norm(previous)
    return r


def reconstruct(ls, P_lambda_squared_observed, ls_obs_min, ls_obs_max, 
                mu=0.01, phi_max=500, rtol = 0.001, max_iter=1000):
    """
    Core of the CRAFT introduced in Cooray et al. 2020b. 
    
    Takes in the list of lambda squared sampling and the observed linear 
    polarization spectrum to return the reconstructed polarization 
    spectrum. Input the observed spectrum with zero at unobserved lambda 
    squared values. 
    
    Parameters
    ----------
    ls  : ndarray
        Regularly sampled array of lambda squared of the observed 
        linear polarization spectrum.
    P_lambda_squared_observed : ndarray
        Observed complex linear polarization spectrum.
    ls_obs_min : float
        Lower limit of lamdba squared in the observed spectrum
    ls_obs_max : float
        Upper limit of lamdba squared in the observed spectrum
    mu : float (optional)
        Parameter of the nonlinear threshold operator S_mu. 
        Default set to 0.01 
    phi_max : positive float (optional)
        Parameter to set the initial window in Faraday depth.
        The parameter is the maximum absolute Faraday depth phi.
        Default set to 500
    rtol : float (optional)
        The covergence criterion by relative tolerance.
        Default set to 0.001
    max_iter : int (optional)
        The covergence criterion by relative tolerance.
        Default set to 1000
    
    Returns
    -------
    P_lambda_squared_reconstructed : ndarray
        Reconstructed complex linear polarization spectrum with same 
        lambda square sampling.
    
    """

    phi, F_phi_obs = IFT_1D(ls, P_lambda_squared_observed)
    dphi = phi[1] - phi[0]

    g = P_lambda_squared_observed.copy()
    g_n = g
    F_f_0 = IFT_1D(ls, g)[1]
    F_f_n = np.copy(F_f_0)
    
    F_previous = np.ones(F_f_0.shape)
    F_current = F_f_0
    
    recon_scale = chi_smoothing_scale(ls_obs_min, ls_obs_max)

    F_phi_window = np.ones(phi.shape)
    F_phi_window[np.abs(phi)<phi_max] = 0

    n=0
#     ls2 = ls.copy()
    while normalized_residual(F_current, F_previous)>rtol and n<max_iter:

        g_n = np.add(np.multiply(g_n,ls_mask),
                     np.multiply(g,ls_wind))

        F_f_n = IFT_1D(ls, g_n)[1]
    #     phi2, F_f_n = IFT_1D(ls2, g_n)#[1]

        F_phi_window_new[np.abs(F_f_n)<mu] = 0
        F_f_n = np.multiply(F_f_n, F_phi_window)

        abs_FDF = np.abs(F_f_n)
        chi = np.angle(F_f_n)

        chi[F_phi_window==0] = 0
        abs_FDF[abs_FDF>mu] -= mu

        chi = ndimage.gaussian_filter1d(chi, (recon_scale/2/0.1), 
                                        mode='constant', cval=0)

        F_f_n = abs_FDF*np.cos(chi) + 1j*abs_FDF*np.sin(chi)

        g_n = FT_1D(phi, F_f_n)[1]
    #     ls2, g_n = FT_1D(phi2, F_f_n)

        F_previous = F_current.copy()
        F_current = F_f_n
        n+=1

    # print(n)
    
    P_lambda_squared_reconstructed = FT_1D(phi, F_f_n)[1]
    
    return P_lambda_squared_reconstructed

##############################



