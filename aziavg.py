import numpy as np

def aziavg(data, center=None, rad=None, weight=None, deinterlace=0):
    """Azimuthal average

    Args:
       data (array): Two-dimensional data array.

    Kwargs:
       center: [xc, yc] coordinates of the point around which to compute average
           Default: geometric center

       rad (int): maximum radius of average [pixels]
           Default: half of the minimum dimension of the data

       weight (array): relative weighting of each pixel in data.
           Default: uniform weighting

    Returns:
        array: One-dimensional azimuthal average

    """
    y, x = np.indices(data.shape)

    if center is None:
        xc = (x.max() - x.min())/2.
        yc = (y.max() - y.min())/2.
    else:
        xc, yc = center

    if rad is None:
        rad = np.min([xc, x.max() - xc - 1, yc, y.max() - yc - 1])
    rad = np.floor(rad).astype(int) + 1

    a = data
    if weight is not None:
        a *= weight
            
    # distance to center
    r = np.hypot(x - xc, y - yc)

    if deinterlace > 0:
        n = np.mod(deinterlace, 2)
        a = a[:,n::2]
        r = r[:,n::2]

    # bin by distance to center
    rn = np.arange(rad)
    ndx = np.digitize(r.flat, rn)

    # apportion data according to position in bin
    fh = r - np.floor(r)
    fl = 1. - fh
    ah = fh * a
    al = fl * a
    
    # bin up data according to distance
    acc = np.zeros(rad)
    count = np.zeros(rad)
    for n in xrange(1, rad):
        w = (ndx == n)
        acc[n-1] += al.flat[w].sum()
        acc[n] = ah.flat[w].sum()
        count[n-1] += fl.flat[w].sum()
        count[n] = fh.flat[w].sum()
        
    return acc/np.maximum(count,1e-3)

if __name__ == "__main__":
    y, x = np.indices([101,101])
    yc, xc = 40, 40
    r = np.hypot(x-xc, y-yc)
    a = aziavg(r, center = [xc, yc])
