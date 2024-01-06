import numpy as np

draw_shape = np.array([[-15,0],[15,0],[0,15]])

angle = np.pi/2
rot_mat = np.array([[np.cos(angle), np.sin(angle)],\
                    [-np.sin(angle), np.cos(angle)]])

new_shape_points = np.dot(draw_shape, rot_mat)

print(new_shape_points)