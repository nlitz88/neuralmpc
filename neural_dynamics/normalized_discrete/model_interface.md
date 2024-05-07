# Normalized Discrete Learned Model
At a high level, this model accepts the vehicle's state and current controls as
input and outputs the vehicle's state at the next timestep.

## Model Interface
See the glossary section for each term's meaning and what it expands to.


**NOTE**: This describes the interface of the entire torchscript model--not the
neural network itself. These are the inputs and outputs you need to know if you
are using this model. If you need to understand the actual transformations +
normalization being done to the inputs and outputs prior to being passed into
the neural network, check out the [nn_interface](./nn_interface.md) document.

### Model Input:
`[u_k, x_k]`

This expands to the following:

`[u_k, q_dot_B_k, q_N_k]`

Which further expands to:

`[u_a_k, u_steer_k, vx_B_k, vy_B_k, w_B_k, xp_N_k, yp_N_k, phi_B_N_k]`

Which maps to the following in terms of our message names:

`[u/u_a, u/u_steer, v/v_long, v/v_tran, w/w_psi]`

### Model Output:

`[x_k_1]`

This expands to the following:

`[q_dot_B_k_1, q_N_k_1]`

Which expands further to:

`[vx_B_k_1, vy_B_k_1, w_B_k_1, xp_N_k_1, yp_N_k_1, phi_B_N_k_1]`

### Term Glossary
Term convention: `component_name_frame_timestep_timestepincrement`
- For example: `xp_N_k_1` == "The x-position in the inertial frame at timestep
  k+1"

Control Inputs
- `u`: Array of control inputs. Expands to `u_a, u_steer`.
- `u_a`: Acceleration control input. Maps to "u/u_a" in our log files.
- `u_steer`: Steering control input. Maps to "u/u_steer" in our log files.

Frames:
- `N`: The inertial/world/global frame.
- `B`: The body/ego frame.

State:
- `x`: The car's state vector. In general, the state vector contains the car's
  configuration `q` and the rate of change of that configuration `q_dot`. There
  contains `q, q_dot`. 
- `q`: The car's configuration. In general, the configuration contains the car's
  pose == position and (relative) orientation (== attitude). `q` contains
  `xp, yp, phi`. By convention, the configuration is *usually* expressed in the
  inertial frame `N`.
- `xp`: The car's position in the x direction of whatever frame it is expressed
  in. Maps to "x/x" in our logs.
- `yp`: The car's position in the y direction of whatever frame it is expressed
  in. Maps to "x/y" in our logs.
- `phi`: The car's relative rotation (attitude) between two frames. Usually,
  this is the rotation from the body frame to the inertial frame. Maps to
  "e/psi" in our logs.
- `q_dot`: The rate of change of the car's configuration. By convention, `q_dot`
  is usually expressed in the body frame. Expands to / contains `vx, vy, w`.
- `vx`: Velocity along the x-direction of whatever frame it is expressed in. By
  convention, this is usually the body frame, making this the body-longitudinal
  velocity. Maps to "v/v_long" in our logs.
- `vy`: Velocity along the x-direction of whatever frame it is expressed in. By
  convention, this is usually the body frame, making this the body-lateral
  velocity. Maps to "v/v_tran" in our logs.
- `w`: Angular velocity about the z-axis of whatever frame it is expressed in. By
  convention, this is usually the body frame. Maps to "w/w_psi" in our logs.

## Background/Rationale
Per Zac Manchester's Lecture: Conventionally in robotics, a robot's state **x**
is composed of the robot's configuration (q)--which is itself compose of the
robot's position and orientation (together called "pose")--as well as the rate
of change of that configuration (q_dot).

The frame that each of these components (q and q_dot) are expressed in is a
matter of convention. The convention in robotics is:
- The robot's q is usually expressed in terms of the inertial frame.
- The robot's q_dot is usually expressed in terms of the body/ego frame.
