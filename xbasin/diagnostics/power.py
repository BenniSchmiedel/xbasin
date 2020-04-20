import xarray as xr
from .transport import Transport


class Power:

    def __init__(self,
                 grid_ops,
                 position_out='T',
                 interpolation_step='consecutive',
                 ):

        self.position_out = position_out
        self.interpolation_step = interpolation_step
        self.ops = grid_ops
        self.transport = Transport(grid_ops)

    def P_taug(self, tau_x, tau_y, eta, f, rho_0=1013.25, g=9.81, position_out=None, f_interpolation=False, **kwargs):
        """
        Computes the direct rate of power input into the geostrophic circulation.

        It is calculated from the surface wind stress (tau_x, tau_y) and the sea surface height eta with:
        P_tau = rho_0 * g * [ U_ek * Grad(eta) ] = rho_0 * g * [ ( (k x tau) / (g * f) ) * Grad(eta) ]

        Computation is performed with interpolation of the data variables to give the result on a specified
        grid position.
        Note: The windstress z-coordinate has a default 0-input to stay consistent with the crossproduct operator.

        When interpolation is done preceding to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau {'U' and 'V'} --interp--> 'T' ===>>> U_Ek {['T','T',__}}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['T','T', __ ]}
        ===>>> P_tau {'T'}

        'F'-point:
        tau {'U' and 'V'} --interp--> 'F' ===>>> U_Ek {['F','F',__}}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['F','F', __ ]}
        ===>>> P_tau {'F'}

        When interpolation is done consecutive to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['T','T', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['T','T', __ ]}
        ===>>> P_tau {'T'}

        'F'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['F','F', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['F','F', __ ]}
        ===>>> P_tau {'F'}

        :param tau_x:           wind stress x-direction
        :param tau_y:           wind stress y-direction
        :param eta:             sea surface height eta
        :param f:               coriolis parameter
        :param rho_0:           surface water density
        :param g:               gravitational acceleration
        :param position_out:    position on grid of output P_taug

        :return:                P_taug
        """
        tr = self.transport
        tau = [tau_x, tau_y, 0]
        ax_order = ('X', 'Y', 'Z')

        if position_out is None:
            position_out = self.position_out

        # Get positions of elements
        #(tau_pos, eta_pos) = (self.ops._get_position(tau), self.ops._get_position(eta))

        if self.interpolation_step == 'preceding':
            # Interpolate tau to position_out
            tau=self.ops._shift_position(tau,position_out,elements=[True,True,False])
            #Calculate U_ek
            U_Ek = self.transport.ekman_transport(tau, f, position_out=position_out, f_interpolation=f_interpolation)
            # Compute Grad(eta), fill z-coordinate with 0
            grad_eta = []
            for i in range(2):
                grad_eta.append(self.ops.derivative(eta, axis=ax_order[i], **kwargs))
            grad_eta.append(0)
            # Interpolate Grad(eta) to position_out
            grad_eta=self.ops._shift_position(grad_eta,position_out,elements=[True,True,False])


        elif self.interpolation_step == 'consecutive':
            # Calculate U_ek
            U_Ek = self.transport.ekman_transport(tau, f, f_interpolation=f_interpolation)

            print([U_Ek[i].dims for i in range(2)])
            # Interpolate tU_ek to position_out
            U_Ek = self.ops._shift_position(U_Ek, position_out, elements=[True, True, False])
            # Compute Grad(eta), fill z-coordinate with 0
            grad_eta = []
            for i in range(2):
                grad_eta.append(self.ops.derivative(eta, axis=ax_order[i], **kwargs))
            grad_eta.append(0)
            # Interpolate Grad(eta) to position_out
            grad_eta = self.ops._shift_position(grad_eta, position_out, elements=[True, True, False])



        p_tau = rho_0 * g * self.ops.dot(U_Ek, grad_eta)

        return p_tau

    def P_down(self):

        return
