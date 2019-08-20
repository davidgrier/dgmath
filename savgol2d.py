# +
# NAME:
#    savgol2d()
#
# PURPOSE:
#    Generate two-dimensional Savitzky-Golay smoothing and derivative kernels
#
# CALLING SEQUENCE:
#    filter = savgol2d(dim, order)
#
# INPUTS:
#    dim: width of the filter [pixels]
#    order: The degree of the polynomial
#
# KEYWORD PARAMETERS:
#    dx: order of the derivative to compute in the x direction
#        Default: 0 (no derivative)
#    dy: order of derivative to compute in the y direction
#        Default: 0 (no derivative)
#
# OUTPUTS:
#    filter: [dim, dim] Two-dimensional Savitzky-Golay filter
#
# EXAMPLE:
# IDL> dadx = convol(a, savgol2d(11, 6, dx = 1))
#
# MODIFICATION HISTORY:
#  Algorithm based on SAVGOL2D:
#  Written and documented
#  Fri Apr 24 13:43:30 2009, Erik Rosolowsky <erosolo@A302357>
#
#  02/06/2013 Written by David G. Grier, New York University
#  09/2013 Translated to Python by Mark D. Hannel, New York University
#  Copyright (c) 2013 David G. Grier
# -

import numpy as np
from itertools import count


def savgol2d(dim, order, dx=0, dy=0):
    """
    Generates two-dimensional Savitzky-Golay smoothing and derivative kernels

   return filter.reshape(dim, dim)
