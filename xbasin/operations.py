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

    def __init__(self,
                 grid,
                 discretization='standard',
                 boundary={'boundary': 'fill', 'fill_value': 0}):
        """
        Creates the standard configuration for the operations used.
        The configuration of an operation can be customized by will if required.
        """
        self.grid = grid
        self.discretization = discretization
        self.boundary = boundary
        # Position types, hard coded
        self.points = {'T': (('z_c', 'y_c', 'x_c'), ('y_c', 'x_c'), ('z_c',)),
                       'U': (('z_c', 'y_c', 'x_f'), ('y_c', 'x_f'), ('x_f',)),
                       'V': (('z_c', 'y_f', 'x_c'), ('y_f', 'x_c'), ('y_f',)),
                       'F': (('z_c', 'y_f', 'x_f'), ('y_f', 'x_f')),
                       'W': (('z_f', 'y_c', 'x_c'), ('z_f',)),
                       'UW': (('z_f', 'y_c', 'x_f'),),
                       'VW': (('z_f', 'y_f', 'x_c'),),
                       'FW': (('z_f', 'y_f', 'x_f'),)
                       }
    def _data_skip(self,da,skip):
        """
        Returns a cut version of data vector da by skipping physical dimensions


        """
        if type(skip) is int:
            da = da[:skip] + da[skip + 1:]
        elif type(skip) is list:
            k = 0
            for ind in skip:
                da = da[:ind - k] + da[ind + 1 - k:]
                k += 1
        else:
            raise Exception("""Input for "skip" not understood. Give integer or list.""")
        return da
    def _get_dims(self, da):
        """
        Returns the spatial position of the data variable of data vector.
        Note: Output does not include 't'-dimension

        :param da:  data variable or vector as xarray.Dataarray or List(xarray.Dataarray, ...)

        :return:    position of data varaible or vector
        """

        # If da is a list (vector), return a list with the dimension tuple for every direction.
        if type(da) == list:
            pos = []
            for i in range(len(da)):
                dims = da[i].dims
                # Remove temporal dimension if present
                if 't' in dims:
                    ind = dims.index('t')
                    dims = dims[:ind] + dims[ind + 1:]
                pos.append(dims)
        else:
            dims = da.dims
            if 't' in dims:
                ind = dims.index('t')
                dims = dims[:ind] + dims[ind + 1:]
            pos = dims

        return pos

    def _get_position(self, da, skip=None):
        """
        Returns the spatial position of the data variable of data vector.
        Note: Output does not include 't'-dimension

        :param da:  data variable or vector as xarray.Dataarray or List(xarray.Dataarray, ...)

        :return:    position of data varaible or vector
        """

        # If da is a list (vector), return a list with the dimension tuple for every direction.
        if skip != None:
            da = self._data_skip(da, skip)
        dims = self._get_dims(da)
        if type(dims) is list:
            positions=[]
            for dim in dims:
                for pos in self.points:
                    if dim in self.points[pos]:
                        positions.append(pos)
                        break
            if len(positions) != len(dims):
                raise Exception("""Could not get an appropriate grip point position for the given dimensions:
                                    %s """ % (dims,))
            return positions
        else:
            position=None
            for pos in self.points:
                if dims in self.points[pos]:
                    position = pos
                    break
            if position is None:
                raise Exception("""Dimension does not match any know grip point position: %s """ % (dims, ))
            return position



    def _get_missmatch(self, da, pos):
        """
        Returns the axes-name where a missmatch between the variable and a position is found.
        The axes-names are hard coded to 'X', 'Y', or 'Z'. Has to match the name given for the xgcm metric!

        :param da:  data variable
        :param pos: position where it is compared to ('T','U','V','F','W','UW','VW','FW')

        :return:    List of axes-names of missmatches
        """
        # Get dimensions, if temporal dimension cut it from dims to get spatial dims only
        dims = da.dims
        if 't' in dims:
            ind = dims.index('t')
            dims = dims[:ind] + dims[ind + 1:]

        # Get the expected dimension with corresponding number of axes
        ax_num = [len(l) for l in self.points[pos]].index(len(dims))
        expect = self.points[pos][ax_num]

        # Add physical dimensions where a missmatch is found
        missmatch = []
        for ax in range(len(expect)):
            if dims[ax] == expect[ax]:
                pass
            else:
                ax_miss = 'X' if dims[ax][0] == 'x' else \
                        'Y' if dims[ax][0] == 'y' else \
                        'Z' if dims[ax][0] == 'z' else None
                if ax_miss == None:
                    raise Exception("""Axis %s does not match to any known dimensions""" % (da.dims[ax]))
                missmatch.append(ax_miss)

        return missmatch

    def _shift_position(self, da, output_position, elements=None):
        """
        Returns the variable interpolated to a prescribed output position.

        :param da:                  data variable or vector to shift
        :param output_position:     desired output position
        :param elements:            elements to include in the shift if a data vector is given, List(Boolean, ..)

        :return:                    interpolated data variable or vector onto the output_position
        """

        # If da is a list (vector), interpolate every element in da along the axes to match the output_position
        # If elements is given only indices where True is given are included. Default is True for every element
        if type(da) == list or type(da) == np.ndarray:
            if elements is None:
                elements = [True] * len(da)
            if type(output_position) is str:
                output_position = [output_position] * len(da)
            da_out = []
            for i in range(len(da)):
                if elements[i]:
                    element = da[i]
                    pos = output_position[i]
                    if self._matching_pos(element, pos):
                        da_out.append(element)
                    else:
                        missmatch = self._get_missmatch(element, pos)
                        da_out.append(self.interp(element, axis=missmatch))
                else:
                    da_out.append(da[i])
        # Else interpolate the data variable along the axes to match the output_position
        else:
            if self._matching_pos(da, output_position):
                da_out = da
            else:
                missmatch = self._get_missmatch(da, output_position)
                da_out = self.interp(da, axis=missmatch)

        return da_out

    def _matching_pos(self, da, pos, skip=None):
        """
        Checks if the given data is on the indicated position

        :param da:      data variable or vector to test
        :param pos:     position to test for

        :return:        True, False or List(True/False, ...)
        """

        if type(da) == list:
            if skip != None:
                da = self._data_skip(da,skip)
            a_pos = self._get_dims(da)

            match = [any([expect == element for expect in self.points[pos]]) for element in a_pos]
            if all(match):
                return True
            else:
                return match  # raise Exception("""False elements %s do not match position %s""" % (match, pos))
        else:
            a_pos = self._get_dims(da)
            if a_pos in self.points[pos]:
                return True
            else:
                return False  # raise Exception("""The variable does not match position %s""" %pos)

    def _matching_dim(self, da1, da2, skip1=None, skip2=None):
        """
        Checks if the dimension of data variable da1 matches the dimension of data variable da2.

        :param da:      data variable or vector to test
        :param pos:     position to test for

        :return:        True, False or List(True/False, ...)
        """
        # Get dimensions and cut 't' and skip if given
        if skip1 != None:
            da1 = self._data_skip(da1,skip1)
        if skip2 != None:
            da2 = self._data_skip(da2,skip2)

        da1_dims = self._get_dims(da1)
        da2_dims = self._get_dims(da2)
        if 't' in da1_dims:
            ind = da1_dims.index('t')
            da1_dims = da1_dims[:ind] + da1_dims[ind + 1:]
        if 't' in da2_dims:
            ind = da2_dims.index('t')
            da2_dims = da2_dims[:ind] + da2_dims[ind + 1:]


        # Depending on type, make comparison
        if type(da1) is list and type(da2) is xr.DataArray:
            match = [da2_dims == da1_pos for da1_pos in da1_dims]
            if all(match):
                return True
            else:
                return match  # raise Exception("""False elements %s do not match position %s""" % (match, pos))
        elif type(da2) is list and type(da1) is xr.DataArray:
            match = [da1_dims == da2_pos for da2_pos in da2_dims]
            if all(match):
                return True
            else:
                return match
        elif type(da1) is list and type(da2) is list:
            if len(da1) == len(da2):
                if da1_dims == da2_dims:
                    return True
                else:
                    return False
            else:
                raise Exception("""Data variable 1 and Data variable 2 do not have the same length""")
        elif type(da1) is xr.DataArray and type(da2) is xr.DataArray:
            if da1_dims == da2_dims:
                return True
            else:
                return False

    def derivative(self, da, axis, **kwargs):

        if self.discretization == 'standard':
            da_der = self.grid.derivative(da, axis, **self.boundary, **kwargs)

        return da_der

    def interp(self, da, axis, **kwargs):
        """
        Interpolation function based on the xgcm interpolation.
        Can be performed for up to three axes.

        :param da:      data variable to interpolate
        :param axis:    axis or List(axes) along which to interpolate, computed by given order
        :param kwargs:  additional keyword arguments for xgcm.interp

        :return:        interpolated data variable along given axis
        """
        if self.discretization == 'standard':
            if axis == None:
                da_int = self.grid.interp(da, **self.boundary, **kwargs)
            elif len(axis) == 1:
                da_int = self.grid.interp(da, axis=axis[0], **self.boundary, **kwargs)
            elif len(axis) == 2:
                da_int = self.grid.interp(self.grid.interp(da, axis=axis[0], **self.boundary, **kwargs),
                                          axis=axis[1], **self.boundary, **kwargs)
            elif len(axis) == 3:
                da_int = self.grid.interp(self.grid.interp(self.grid.interp(da,
                                                                            axis=axis[0], **self.boundary, **kwargs),
                                                           axis=axis[1], **self.boundary, **kwargs),
                                          axis=axis[2], **self.boundary, **kwargs)

            else:
                raise Exception("Unknown operation for axis: %s" % (axis))

        return da_int

    def dot(self, x, y):
        """
        Performs a dot product for two vector fields.

        x and y are vector fields given as a list of scalar fields. Matching dimensions are required.

        :param x:   vector field
        :param y:   vector filed

        :return:
        """
        dims = len(x) if len(x) == len(y) else False
        if dims == False:
            raise Exception("Vector dimensions do not match: x: %s != y:%s" % (len(x), len(y)))

        c = 0
        for i in range(dims):
            c += x[i] * y[i]

        return c

    def cross(self, x, y, position_out=(None, None, None)):
        """
        Performs a cross product for two 3d vector fields.

        x and y are vector fields as a type 'list' or 'np.ndarray" of scalar fields. Matching dimensions are required.

        :param a:
        :param b:
        :return:
        """

        dims = len(x) if len(x) == len(y) else False
        if dims == False:
            raise Exception("Vectors dimensions do not match: x: %s != y:%s" % (len(x), len(y)))

        if dims != 3:
            raise Exception("Operation is only implemented for 3D vectors, but dimensions are: %s" %dims)

        c = [0] * 3
        c[0] = x[1] * y[2] - x[2] * y[1]
        c[1] = x[2] * y[0] - x[0] * y[2]
        c[2] = x[0] * y[1] - x[1] * y[0]

        if type(x) == np.ndarray and type(y) == np.ndarray:
            c = np.ndarray(c)
        return c
