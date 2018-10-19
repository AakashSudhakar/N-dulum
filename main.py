"""
FILENAME:       main.py
TITLE:          The N-dulum Project. 
AUTHOR:         Aakash Sudhakar
DESCRIPTION:    A physics-based simulation in Python attempting to codify and 
                animate an Nth-order pendulum based on user input and advanced
                differential mathematics.

Credit goes to Jake VanderPlas, Gilbert Gede, and user @christian on SciPython 
for composing beautiful Python projects tackling similar complex pendulum-based 
simulations. Your work was invaluable to my understanding and implementation 
of the N-dulum project. 

Special thanks to Alan Davis, Mike Kane, Milad Toutounchian, and many other 
instructors and fellow students at the Make School Product College for their 
feedback, suggestions, and continued support. 

(C) October 2018
"""

################################################################################
###### IMPORT STATEMENTS FOR SIMULATION MODELING AND ADVANCED MATHEMATICS ######
################################################################################


import sys                                      # Interpreter-Level Functionality
import numpy as np                              # Advanced Numerical Mathematics
import matplotlib.pyplot as plt                 # Data Visualization Toolkit
from matplotlib import animation, collections
from pylab import *                             # Extended SciPython Toolkit
from sympy import symbols, Dummy, lambdify      # Advanced Symbolic Mathematics
from sympy.physics import mechanics
from scipy.integrate import odeint
from random import randrange                    # Random Integer Function
from IPython.display import HTML                # Inner HTML Video Viewer
from time import time                           # Modular Runtime Tracker


################################################################################
###### IMPORT STATEMENTS FOR SIMULATION MODELING AND ADVANCED MATHEMATICS ######
################################################################################


class Nth_Order_Pendulum_Simulator(object):
    
    # COMPLETE? NEED TO BE CAREFUL WITH INSTANCE VARIABLES
    def __init__(self, N):
        self.N = N                                      # Required input argument
        self.time_vector = np.linspace(0, 10, 1000)     # Defaults to linspace() arg
        self.pos_init = 235                             # Defaults to int(235)
        self.vel_init = 0                               # Defaults to int(0)
        self.mass_vector = 1                            # Defaults to int(1)
    
    # COMPLETE
    def integrate_pendulum_odes(self, adv_time_vector=None, pos_init=235):
        """
        Method to integrate Nth-order pendulum ODEs.
        """
        
        # Instantiates physics constants for position, velocity, mass, length, gravity, and time
        q, u = mechanics.dynamicsymbols("q:{0}".format(str(self.N))), mechanics.dynamicsymbols("u:{0}".format(str(self.N)))
        m, l = symbols("m:{0}".format(str(self.N))), symbols("l:{0}".format(str(self.N)))
        g, t = symbols("g t")

        # Creates intertial reference frame
        frame = mechanics.ReferenceFrame("frame")
        
        # Creates focus point in intertial reference frame
        point = mechanics.Point("point")
        point.set_vel(frame, 0)
        
        # Instantiates empty objects for pendulum segment points, reactionary forces, and resultant kinematic ODEs
        particles, forces, kin_odes = list(), list(), list()

        # Iteratively creates pendulum segment kinematics/dynamics objects
        for iterator in range(self.N):
            # Creates and sets angular velocity per reference frame for each pendulum segment
            frame_i = frame.orientnew("frame_{}".format(str(iterator)), "Axis", [q[iterator], frame.z])
            frame_i.set_ang_vel(frame, u[iterator] * frame.z)

            # Creates and sets velocity of focus point for each pendulum segment
            point_i = point.locatenew("point_{}".format(str(iterator)), l[iterator] * frame_i.x)
            point_i.v2pt_theory(point, frame, frame_i)

            # Creates reference point for each pendulum segment
            ref_point_i = mechanics.Particle("ref_point_{}".format(str(iterator)), point_i, m[iterator])
            particles.append(ref_point_i)

            # Creates objects for reference frame dynamics
            forces.append((point_i, m[iterator] * g * frame.x))
            kin_odes.append(q[iterator].diff(t) - u[iterator])
            point = point_i

        # Generates position and velocity equation systems using Kane's Method
        kane = mechanics.KanesMethod(frame, q_ind=q, u_ind=u, kd_eqs=kin_odes)
        fr, frstar = kane.kanes_equations(particles, forces)

        # Creates vector for initial positions and velocities
        y0 = np.deg2rad(np.concatenate([np.broadcast_to(pos_init, self.N), 
                                        np.broadcast_to(self.vel_init, self.N)]))

        # Creates vectors for pendulum segment lengths and masses
        length_vector = np.ones(self.N) / self.N
        length_vector = np.broadcast_to(length_vector, self.N)
        self.mass_vector = np.broadcast_to(self.mass_vector, self.N)

        # Instantiates and creates fixed constant vectors (gravity, lengths, masses)
        params = [g] + list(l) + list(m)
        param_vals = [9.81] + list(length_vector) + list(self.mass_vector)

        # Initializes objects to solve for unknown parameters
        dummy_params = [Dummy() for iterator in q + u]
        dummy_dict = dict(zip(q + u, dummy_params))

        # Converts unknown parametric objects into Kane's Method objects for numerical substitution
        kin_diff_dict = kane.kindiffdict()
        mass_matrix = kane.mass_matrix_full.subs(kin_diff_dict).subs(dummy_dict)
        full_forcing_vector = kane.forcing_full.subs(kin_diff_dict).subs(dummy_dict)

        # Functionalizes Kane's Method unknown parametric objects using NumPy for numerical substitution
        mm_func = lambdify(dummy_params + params, mass_matrix)
        ff_func = lambdify(dummy_params + params, full_forcing_vector)

        # Defines helper method with gradient calculus to use with ODE integration
        def __parametric_gradient_function(y, t, args):
            """
            Helper function to derive first-order equations of motion from parametric arguments.
            """
            values = np.concatenate((y, args))
            solutions = np.linalg.solve(mm_func(*values), ff_func(*values))
            return np.array(solutions).T[0]
        
        if adv_time_vector is None:
            return odeint(__parametric_gradient_function, y0, self.time_vector, args=(param_vals,))
        else:
            return odeint(__parametric_gradient_function, y0, adv_time_vector, args=(param_vals,))
    
    # COMPLETE
    def visualize_timewise_displacement(self, ap_vector):
        """
        Method to visualize positional displacement over time.
        """        
        fig, ax = subplots(2, sharex=True, sharey=False)
        fig.set_size_inches(6.5, 6.5)

        for iterator in range(self.N):
            ax[0].plot(self.time_vector, ap_vector[:, iterator], label="$q_{}$".format(iterator))
            ax[1].plot(self.time_vector, ap_vector[:, iterator + self.N], label="$u_{}$".format(iterator))

        ax[0].legend(loc=0)
        ax[1].legend(loc=0)
        ax[1].set_xlabel("Time (s)")
        ax[0].set_ylabel("Angle (rad)")
        ax[1].set_ylabel("Angular rate (rad/s)")

        fig.subplots_adjust(hspace=0)
        setp(ax[0].get_xticklabels(), visible=False)
        tight_layout()
        show()
        return
    
    # COMPLETE
    def get_xy_displacement(self, ap_vector, to_viz=False):
        """
        Method to get positional displacement in coordinate-matrix or visualization form.
        """
        ap_vector = np.atleast_2d(ap_vector)
        n = ap_vector.shape[1] // 2
        length_vector = np.ones(n) / n

        zeros_vector = np.zeros(ap_vector.shape[0])[:, None]
        x = np.hstack([zeros_vector, length_vector * np.sin(ap_vector[:, :n])])
        y = np.hstack([zeros_vector, -length_vector * np.cos(ap_vector[:, :n])])

        plt.xlabel("X-Position")
        plt.ylabel("Y-Position")

        if to_viz is False:
            return np.cumsum(x, 1), np.cumsum(y, 1)
        else:
            plt.plot(np.cumsum(x, 1), np.cumsum(y, 1))
            show()
            return
    
    # INCOMPLETE
    def animate_nth_order_pendulum(self, tracer_length=None, to_save=False):
        ap_vector = self.integrate_pendulum_odes()
        x, y = self.get_xy_displacement(ap_vector)

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("off")
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

        line, = ax.plot(list(), list(), "o-", lw=2)

        def _init_():
            line.set_data(list, list)
            return line,

        def animate_(iterator):
            line.set_data(x[iterator], y[iterator])
            return line,

        anim_obj = animation.FuncAnimation(fig, animate_, frames=len(self.time_vector),
                                           interval=1000 * self.time_vector.max() / len(self.time_vector),
                                           blit=True, init_func=_init_)
        plt.close(fig)
        
        if to_save is True:
            curr_anim_loc = "animations/single/pendulum_model_order-{}_single.mp4".format(self.N)
            anim_obj.save(curr_anim_loc)
        return anim_obj
    
    # INCOMPLETE
    def animate_multiple_pendulums_with_tracers(self, number_of_pendulums=12, perturbation=1E-6, tracer_length=15, to_save=False):
        oversample = 3
        tracer_length *= oversample
        adv_time_vector = np.linspace(0, 10, oversample * 200)
        
        ap_vector = [self.integrate_pendulum_odes(adv_time_vector, pos_init=135+iterator*perturbation/number_of_pendulums)
                    for iterator in range(number_of_pendulums)]
        pos_vector = np.array([self.get_xy_displacement(pos) for pos in ap_vector])
        pos_vector = pos_vector.transpose(0, 2, 3, 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

        tracer_segments = np.zeros((number_of_pendulums, 0, 2))
        tracer_collection = collections.LineCollection(tracer_segments, cmap="rainbow")
        tracer_collection.set_array(np.linspace(0, 1, number_of_pendulums))
        ax.add_collection(tracer_collection)

        points, = plt.plot(list(), list(), "ok")

        pendulum_segments = np.zeros((number_of_pendulums, 0, 2))
        pendulum_collection = collections.LineCollection(pendulum_segments, colors="black")
        ax.add_collection(pendulum_collection)

        def _init_():
            pendulum_collection.set_segments(np.zeros((number_of_pendulums, 0, 2)))
            tracer_collection.set_segments(np.zeros((number_of_pendulums, 0, 2)))
            points.set_data(list(), list())
            return pendulum_collection, tracer_collection, points

        def animate_(iterator):
            iterator *= oversample
            pendulum_collection.set_segments(pos_vector[:, iterator])
            tracer_slice = slice(max(0, iterator - tracer_length), iterator)
            tracer_collection.set_segments(pos_vector[:, tracer_slice, -1])
            x, y = pos_vector[:, iterator].reshape(-1, 2).T
            points.set_data(x, y)
            return pendulum_collection, tracer_collection, points

        interval = 1000 * oversample * adv_time_vector.max() / len(adv_time_vector)
        anim_obj = animation.FuncAnimation(fig, animate_, frames=len(adv_time_vector) // oversample,
                                       interval=interval,
                                       blit=True, init_func=_init_)

        plt.close(fig)
        
        if to_save is True:
            curr_anim_loc = "animations/many/pendulum_model_order-{}_many.mp4".format(self.N)
            anim_obj.save(curr_anim_loc)
        return anim_obj


################################################################################
########################### MAIN PROGRAM RUN FUNCTION ##########################
################################################################################


def main():
    triple_pendulum = Nth_Order_Pendulum_Simulator(N=3)
    ap_vector = triple_pendulum.integrate_pendulum_odes()
    # triple_pendulum.visualize_timewise_displacement(ap_vector)
    # triple_pendulum.get_xy_displacement(ap_vector, to_viz=True)
    anim_obj = triple_pendulum.animate_nth_order_pendulum(to_save=True)
    
if __name__ == "__main__":
    main()
    
