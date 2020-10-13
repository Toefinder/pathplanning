import numpy as np 
import pylab as pl
import math
from PRM import is_collision, sample_points, generate_road_map, preprocess_query
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
pl.ion()
np.random.seed(4)
env = environment_2d.Environment(10, 6, 5)
pl.clf()
env.plot()
q = env.random_query()
if q is not None:
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)


sx = 1
sy = 0.5
gx = 4
gy = 1.5
rr = 0.05 # reduce this to have better collision checks

# # Testing is_collision
# print("Testing is_collision")
# print("sx = ", sx)
# print("sy = ", sy)
# print("gx = ", gx)
# print("gy = ", gy)
# print(is_collision(sx, sy, gx, gy, rr))

# Testing sample_points
sample_x, sample_y = sample_points(env.size_x, env.size_y)
for i in xrange(len(sample_x)):
    pl.plot([sample_x[i]], [sample_y[i]], "bo", markersize=5)

# Testing generate_road_map
def plot_road_map(road_map, sample_x, sample_y):
    for i in xrange(len(road_map)):
        for ii in xrange(len(road_map[i])):
            ind = road_map[i][ii]

            pl.plot([sample_x[i], sample_x[ind]], \
                    [sample_y[i], sample_y[ind]], "-k") 

road_map, sample_kd_tree = generate_road_map(sample_x, sample_y, rr)
# plot_road_map(road_map, sample_x, sample_y)

# Testing preprocess_query
sx = x_start
sy = y_start
gx = x_goal
gy = y_goal
full_sample_x, full_sample_y, full_road_map = preprocess_query(sx, sy, gx, gy, rr, \
                                                                sample_x, sample_y, road_map, sample_kd_tree)
plot_road_map(full_road_map, full_sample_x, full_sample_y)
print(full_road_map[-2])
print(full_road_map[-1])

pl.show(block = True)