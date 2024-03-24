__version__ = '0.1.0'

try:
    from igor2 import binarywave
except ImportError:
    import numpy as np
    np.complex = np.complex128 # igor still uses this

    from igor import binarywave
