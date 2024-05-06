import numpy as np
import yaml


class Single_Track_Dynamics():


    def __init__(self, base_config, track_config):
        GRAVITY = 9.81  # Gravity constant


        self.time_step = .01 #MAKE SURE THIS IS CORRECT
        self.m = base_config["ros__parameters"]["chassis"]["total_mass"]
        self.Jzz = base_config["ros__parameters"]["chassis"]["moi"]
        self.l = base_config["ros__parameters"]["chassis"]["wheel_base"]
        self.lr = base_config["ros__parameters"]["chassis"]["cg_ratio"] * self.l
        self.lf = self.l - self.lr
        self.fr = base_config["ros__parameters"]["chassis"]["fr"]
        self.hcog = base_config["ros__parameters"]["chassis"]["cg_height"]

        # Aerodynamics
        self.rho = base_config["ros__parameters"]["aero"]["air_density"]
        self.A = base_config["ros__parameters"]["aero"]["frontal_area"]
        self.cd = base_config["ros__parameters"]["aero"]["drag_coeff"]
        self.cl_f = base_config["ros__parameters"]["aero"]["cl_f"]
        self.cl_r = base_config["ros__parameters"]["aero"]["cl_r"]

        # Tyre
        self.Bf = base_config["ros__parameters"]["front_tyre"]["pacejka_b"]
        self.Cf = base_config["ros__parameters"]["front_tyre"]["pacejka_c"]
        self.Br = base_config["ros__parameters"]["rear_tyre"]["pacejka_b"]
        self.Cr = base_config["ros__parameters"]["rear_tyre"]["pacejka_c"]

        self.kd_f = base_config["ros__parameters"]["front_brake"]["bias"] #TODO: where is this

        self.kb_f = base_config["ros__parameters"]["front_brake"]["bias"]

        self.mu = track_config["ros__parameters"]["single_track_planar"]["mu"]

        self.N = self.m * GRAVITY 

    def body_frame_accel(self, x, u):

        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]

        fd = u[0]
        fb = u[1]
        delta = u[2]



        v_sq = vx * vx

        
        Fx_f = 0.5 * self.kd_f *  fd + 0.5 *  self.kb_f *  fb - 0.5 *  self.fr *  self.N *  self.lr /  self.l
        Fx_fl = Fx_f

        Fx_r = 0.5 * (1 -  self.kd_f) *  fd + 0.5 * (1.0 -  self.kb_f) *  fb - 0.5 *  self.fr *  self.N *  self.lf /  self.l

        ax = ( fd +  fb - 0.5 *  self.cd *  self.rho *  self.A * v_sq -  self.fr *  self.N) / self.m

        Fz_f = 0.5 *  self.N *  self.lr / ( self.lf +  self.lr) - 0.5 *  self.hcog / ( self.lf +  self.lr) *  self.m * ax + 0.25 *  self.cl_f *  self.rho *  self.A * v_sq
        Fz_r = 0.5 *  self.N *  self.lf / ( self.lf +  self.lr) + 0.5 *  self.hcog / ( self.lf +  self.lr) *  self.m * ax + 0.25 *  self.cl_r *  self.rho *  self.A * v_sq

        a_fl =  delta - np.arctan(( self.lf * omega + vy) / (vx + 1e-3))
        a_rl = np.arctan(( self.lr * omega - vy) / (vx + 1e-3))

        Fy_fl =  self.mu * Fz_f * np.sin( self.Cf * np.arctan( self.Bf * a_fl))
        Fy_rl =  self.mu * Fz_r * np.sin( self.Cr * np.arctan( self.Br * a_rl))

        omega_dot = 1.0 /  self.Jzz * (-(2 * Fy_rl) *  self.lr + ((2 * Fy_fl) * np.cos( delta) + (2 * Fx_fl) * np.sin( delta)) *  self.lf)

        vx_dot = 1.0 /  self.m * ((2 * Fx_r) + (2 * Fx_fl) * np.cos( delta) - (2 * Fy_fl) * np.sin( delta) - 0.5 *  self.cd *  self.rho *  self.A * v_sq) + omega * vy
        vy_dot = 1.0 /  self.m * ((2 * Fy_rl) + (2 * Fy_fl) * np.cos( delta) + (2 * Fx_fl) * np.sin( delta)) - omega * vx 

        return vx_dot, vy_dot, omega_dot


    def global_frame_xdot(self,x ):

        phi = x[2]
        vx = x[3]
        vy = x[4]
        omega = x[5]

        px_dot = vx * np.cos(phi) - vy * np.sin(phi)
        py_dot = vx * np.sin(phi) + vy * np.cos(phi)
        phi_dot = omega

        return px_dot, py_dot, phi_dot


    def f_xu(self, x, u):
        '''
        Takes in state (global x, global y, yaw, body vx, body vy, and phi dot)

        Returns derivative of state (global dx, global dy, dyaw, body ax, body ay, a_phi)
        
        '''

        vx_dot, vy_dot, omega_dot = self.body_frame_accel(x, u)

        px_dot, py_dot, phi_dot = self.global_frame_xdot(x)

        return np.vstack((vx_dot, vy_dot, omega_dot, px_dot, py_dot,phi_dot))
    
    def integrate_rk(self, x0, u, dt=None):
        """
        Runge-Kutta Integration. Currently unused
        Integrate initial state x0 (applying constant control u)
        over a time interval of self._time_step, using a time discretization
        of dt.

        :param x0: initial state
        :type x0: np.array
        :param u: control input
        :type u: np.array
        :param dt: time discretization
        :type dt: float
        :return: state after time self._time_step
        :rtype: np.array
        """
        if dt is None:
            dt = 0.25 * self.time_step

        t = 0.0
        x = x0.copy()
        while t < self.time_step - 1e-8:
        
            step = min(dt, self.time_step - t)
   
            # Use Runge-Kutta order 4 integration. For details please refer to
            # https://en.wikipedia.org/wiki/Runge-Kutta_methods.
            u = u.reshape(3,)
            k1 = step * self.f_xu(x, u).reshape(6,)
            k2 = step * self.f_xu(x + 0.5 * k1, u).reshape(6,)
            k3 = step * self.f_xu(x + 0.5 * k2, u).reshape(6,)
            k4 = step * self.f_xu(x + k3, u).reshape(6,)


            delta_x = (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

            # x = self.discrete_step(x, delta_x)
            x += delta_x

     

            t += step

     

        return x
    
    def get_next_state(self, x, u, dt):
        xdot = self.f_xu(x, u)
        x_next = self.integrate_rk(x,u, dt)
        return x_next

def test_functions(dynamics):
    x = np.array([1.0, 1.0, 0.0, .1, -.1, 0.5])
    u = np.array([100.0, 0.0, .1])



    xdot = dynamics.f_xu(x, u)

    x_next = dynamics.integrate_rk(x,u)

    print("xdot",xdot)
    print("x_next",x_next)


if __name__ == '__main__':

    BC_PATH = "/home/adith/Documents/art_racing/Racing-LMPC-ROS2/src/neural_mpc/models/iac_car_base.param.yaml"
    TC_PATH ="/home/adith/Documents/art_racing/Racing-LMPC-ROS2/src/neural_mpc/models/iac_car_single_track.param.yaml"
    with open(BC_PATH, 'r') as file_BC:
        base_config = yaml.safe_load(file_BC)
    with open(TC_PATH, 'r') as file_TC:
        track_config = yaml.safe_load(file_TC)

    test_dynamics =  Single_Track_Dynamics(base_config, track_config)

    test_functions(test_dynamics)







    

        





        
