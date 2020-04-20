import xarray as xr


class Transport:

    def __init__(self, grid_ops, position_out='F', interpolation_step='consecutive'):
        self.position_out = position_out
        self.interpolation_step = 'consecutive'
        self.ops = grid_ops

    def ekman_transport(self, tau, f, rho_0=1013.25, position_out=None, f_interpolation=False):
        """
        Computes the ekman transport from the surface wind stress tau.

        U_ek = rho_o * g * (k x tau)
        with k the vertical unit and tau the windstress.

        :param tau:   windstress vector 3D
        :param rho_0:   surface density
        :param g:   gravitational acceleration
        :param position_out:
        :return:
        """
        # Calculate crossproduct k x tau
        k = [0, 0, 1]
        tau_x_k = self.ops.cross(tau, k)
        if f_interpolation:
            if self.ops._matching_pos(f, position_out):
                pass
            else:
                f = self.ops._shift_position(f, position_out)
        # Calculate U_Ek = tau_x_k /(rho*f), but check if f and tau_x_k have the same dimensions
        if self.ops._matching_dim(tau_x_k, f):
            U_Ek = []
            for ax in range(3):
                if type(f) is list:
                    U_Ek.append(tau_x_k[ax] / (rho_0 * f[ax]))
                else:
                    U_Ek.append(tau_x_k[ax] / (rho_0 * f))
        else:
            raise Exception("""Dimensions of tau x k and f do not match %s - %s
                """ % (self.ops._get_position(tau_x_k), self.ops._get_position(f.dims)))

        return U_Ek
