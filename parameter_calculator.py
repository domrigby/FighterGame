import numpy as np
drag = 0.001

top_speed = 330 # [m/s]
thrust_required = drag*top_speed**2

min_rot_time = 2 # [s]
max_torque = 4*np.pi**(2)*drag/min_rot_time**(2)

print(max_torque)



