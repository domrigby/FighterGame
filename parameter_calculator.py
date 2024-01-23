import numpy as np

area_of_craft = 1
top_speed = 330
air_density = 1.23

thrust_req = 0.5 * top_speed**(2) * air_density

print(thrust_req)

min_turning_circle = 10
ang_drag_to_rudder_len_ratio = min_turning_circle**(2) / 4

print(ang_drag_to_rudder_len_ratio)