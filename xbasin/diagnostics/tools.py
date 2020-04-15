import numpy as np
import xarray as xr
import xgcm

class Grid_ops:
    """
    An object that includes operations for variables defined on a xgcm compatible grid.
    Those operations are defined to simplify dealing with mathematical operations that shift grid point positions
    when applied.

    Note: The operations include approximations of variables onto neighbouring grid positions by interpolation.
    For low grid resolutions the approximation through interpolation might introduce large deviations from the
    original data, especially where strong value shifts are present.
    """

    def __init__(self,discretization='standard'):
        """
        Creates the standard configuration for the operations used.
        The configuration of an operation can be customized by will if required.
        """

        self.discretization = discretization



