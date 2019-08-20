#+
# NAME:
#    circletransform
#
# PURPOSE:
#    Performs an orientational alignment transform, 
#    which is useful for detecting circular features in an image.
#
# CATEGORY:
#    Image analysis, feature detection
#
# CALLING SEQUENCE:
#    b = circletransform(a)
#
# INPUTS:
#    a: [nx, ny] image data
#
# KEYWORD PARAMETERS:
#    deinterlace: if set to an odd number, : only perform
#        transform on odd field of an interlaced image.
#        If set to an even number, transform even field.
#        Default: Not set or set to zero: transform entire frame.
#
# OUTPUTS:
#    b: [nx, ny] circle transform.  Peaks correspond to estimated
#        centers of circular features in a.
#
# PROCEDURE:
#    Compute the gradient of the image.  The local gradient at each
#    pixel defines a line along which the center of a circle may
#    lie.  Cast votes for pixels along the line in the transformed
#    image.  The pixels in the transformed image with the most votes
#    correspond to the centers of circular features in the original
#    image.
#
# REFERENCES:
# 1. F. C. Cheong, B. Sun, R. Dreyfus, J. Amato-Grill, K. Xiao, L. Dixon
#    & D. G. Grier, "Flow visualization and flow cytometry with
#    holographic video microscopy, " Optics Express 17, 
#    13071-13079 (2009)
#
# 2. B. J. Krishnatreya & D. G. Grier, "Fast feature identification
#    for holographic tracking: The orientation alignment transform, "
#    preprint (2013)
#
# EXAMPLE:
#    IDL> b = circletransform(a)
#
# MODIFICATION HISTORY:
# 10/07/2008 Written by David G. Grier, New York University.
# 01/26/2009 DGG Added DEINTERLACE keyword. Gracefully handle
#    case when original image has no features. Documentation cleanups.
# 02/03/2009 DGG Replaced THRESHOLD keyword with NOISE.
# 06/10/2010 DGG Documentation fixes.  Added COMPILE_OPT.
# 05/02/2012 DGG Updated keyword parsing.  Formatting.
# 06/24/2012 DGG Streamlined index range checking in inner loop
#    to improve efficiency.
# 07/16/2012 DGG IMPORTANT: Center results on pixels, not on vertices!
#    Use array_indices for clarity.
# 11/10/2012 DGG Default range should be an integer.
#    Returned array should be cast to integer, not float
# 11/23/2012 DGG Use Savitzky-Golay estimate for derivative.
#    Eliminate SMOOTHFACTOR parameter.  Upgrade parameter checking.
# 11/25/2012 DGG Limit search range by uncertainty in gradient
#    direction.  Remove RANGE keyword.
# 11/30/2012 DGG Optionally return mean range as RANGE keyword.
# 01/16/2013 DGG estimate noise with MAD() by default.
# 01/24/2013 DGG correct test for deinterlace in range(0.
# 02/09/2013 DGG use savgol2d() to compute derivatives.
#    Displace by half a pixel to center, not a whole pixel.
# 02/17/2013 DGG RANGE is the median range of voting pixels, not the
#    mean.
# 03/04/2013 DGG shift by +1 rather than by +0.5.  Limit range if
#    noise is very small.
# 03/17/2013 DGG calculate coordinates explicitly rather than using
#    array_indices, which turns out to be slow.  More efficient array
#    indexing.  No more need to shift pixels for alignment.
# 03/27/2013 DGG eliminate repetitive operations in loops.
# 05/13/2013 DGG suppress borders, which are over-counted.
# 10/04/2013 DGG and Mark Hannel: fix boundary cropping.
# 10/22/2013 DGG added UNCERTAINTY keyword.
# 12/03/2013 DGG Major overhaul: Field-theoretic implementation of
#    the voting algorithm yields factor of 10 speed-up.
# 12/13/2013 DGG use EXTRA for compatibility with pervious version.
# 09/2013 Translated to Python by Mark D. Hannel, New York University
# Copyright (c) 2008-2013 David G. Grier and Mark Hannel
#
#-

#####
# Import Libraries

import numpy as nmp
from edge_convolve import *
from savgol2d import *

# 1) Maybe we should use rfft (real fast fourier transform) for the image
#    and use hfft (hermitian fast fourier transform) for the frequency spectrum
#    NOTE: rfft and hfft only have 1D implementations.  Would have to
#          find a way to
# 2) Maybe the Savitsky-Golay filter should be calculated once,
#    and have it's value stored away in a .dat or .xml file
# 3) Are there easy implementations of fftw in python?



def idl_ifft(dat, axis = None): #citation
    """
    Calculate FFT the same as IDL
    Converting Numpy = [0,99,98,...,1] -----> IDL = [0,1,2,...,99]
    For 2D data, the x and y axes are flipped in a similar way.
    FIXME(JS): Add implementation for 3D data
    """
    
    if dat.ndim == 1:
        dat  = nmp.fft.ifft(dat)
        temp = dat[1:len(dat)].copy()
        dat[1:len(dat)] = temp[::-1]    # reverse array
    elif dat.ndim == 2:
        shape = dat.shape
        if axis == 0:
            dat  = nmp.fft.ifft(dat, axis=axis)
            temp = nmp.flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
        elif axis == 0:
            dat  = nmp.fft.ifft(dat, axis=axis)
            temp = nmp.fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
        else:
            dat  = nmp.fft.ifftn(dat, axes=axis)
            temp = nmp.flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
            temp = nmp.fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
    
    return dat

def idl_fft(dat, axis = None):  #citation
    """
    Calculate FFT the same as IDL
    Converting Numpy = [0,99,98,...,1] -----> IDL = [0,1,2,...,99]
    For 2D data, the x and y axes are flipped in a similar way.
    FIXME(JS): Add implementation for 3D data
    """
    
    if dat.ndim == 1:
        dat = nmp.fft.fft(dat)
        temp = dat[1:len(dat)].copy()
        dat[1:len(dat)] = temp[::-1]    # reverse array
    elif dat.ndim == 2:
        shape = dat.shape
        if axis == 0:
            dat = nmp.fft.fft(dat, axis=axis)
            temp = nmp.flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
        elif axis == 0:
            dat = nmp.fft.fft(dat, axis=axis)
            temp = nmp.fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
        else:
            dat = nmp.fft.fftn(dat, axes=axis)
            temp = nmp.flipud(dat[1:shape[0],:]).copy() # flip x-axis
            dat[1:shape[0],:] = temp
            temp = nmp.fliplr(dat[:,1:shape[1]]).copy() # flip y-axis
            dat[:,1:shape[1]] = temp
    
    return dat



def circletransform(a_, deinterlace = 0):
   """
   Performs an orientational alignment transform, 
   which is useful for detecting circular features in an image.

   INPUTS:
   a: [nx, ny] image data

   OPTIONAL PARAMETERS:
   deinterlace: if set to an odd number, : only perform
     transform on odd field of an interlaced image.
     If set to an even number, transform even field.
     Default: Not set or set to zero: transform entire frame.
     
   OUTPUTS:
   b: [nx, ny] circle transform.  Peaks correspond to estimated
     centers of circular features in a.

   Example:
   >>> b = circletransform(a)
   """
   umsg = 'USAGE: b = circletransform(a)'

   if type(a_) != nmp.ndarray :
      print  umsg 
      return -1

   if a_.ndim != 2 :
      print  umsg 
      print  'A must be a two-dimensional numpy array' 
      return -1

   ny,nx = a_.shape
 

   #dodeinterlace = isa(deinterlace, /scalar, /number) ? deinterlace > 0 : 0
   if deinterlace : # Use to be dodinterlace
      n0 = deinterlace  %2
      a = a_[n0::2, :].astype(float)
      ny = len(a[:, 0])
   else:
      a = a_.astype(float)

   # gradient of image
   # \nabla a = (dadx, dady)
   dx = savgol2d(7, 3, dx = 1)
   dadx = -1*edge_convolve(a, dx) #FIXME: Figure out why edge_convolve returns
                                #   negative answer
   dady = -1*edge_convolve(a, nmp.transpose(dx))

   if deinterlace : dady /= 2.


   # orientational order parameter
   # psi = |\nabla a|**2 \exp(i 2 \theta)
   i = complex(0,1)
   psi = dadx + i*dady ### FIX: May need to swap dadx, dady.
                       ### May also be faster not to use addition
   psi *= psi

   # Fourier transform of the orientational alignment kernel:
   # K(k) = e**(-2 i \theta) / k
   x_row = nmp.arange(nx)/float(nx) - 0.5
   y_col = nmp.arange(ny)/float(ny) - 0.5

   kx,ky = nmp.meshgrid(x_row,y_col)

   if deinterlace : ky /= 2.

   k   = nmp.sqrt(kx**2 + ky**2) + 0.001
   ker = (kx -i*ky)**2 / k**3

   # convolve orientational order parameter with
   # orientational alignment kernel using
   # Fourier convolution theorem
   psi = idl_ifft(psi)
   psi = nmp.fft.fftshift(psi)
   psi *= ker
   psi = nmp.fft.ifftshift(psi)
   psi = idl_fft(psi)

   # intensity of convolution identifies rotationally
   # symmetric centers

   #### Fourth Checkpoint
   
   return nmp.real(psi*nmp.conj(psi))
