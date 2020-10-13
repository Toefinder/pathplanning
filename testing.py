import numpy as np 
import pylab as pl
import math
from PRM import *
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
pl.ion()
np.random.seed(4)
env = environment_2d.Environment(10, 6, 5)
pl.clf()
env.plot()

rr = 0.01 # reduce this to have better collision checks

# # Testing is_collision
# sx = 1
# sy = 0.5
# gx = 4
# gy = 1.5
# print("Testing is_collision")
# print("sx = ", sx)
# print("sy = ", sy)
# print("gx = ", gx)
# print("gy = ", gy)
# print(is_collision(sx, sy, gx, gy, rr))

# # Testing sample_points
# sample_x, sample_y = sample_points(env.size_x, env.size_y)
# for i in xrange(len(sample_x)):
#     pl.plot([sample_x[i]], [sample_y[i]], "bo", markersize=5)

# # Testing generate_road_map
# sample_x, sample_y = sample_points(env.size_x, env.size_y)
# for i in xrange(len(sample_x)):
#     pl.plot([sample_x[i]], [sample_y[i]], "bo", markersize=5)
# road_map, sample_kd_tree = generate_road_map(sample_x, sample_y, rr)
# plot_road_map(road_map, sample_x, sample_y)

# # Testing preprocess_prm
# sample_x, sample_y, road_map, sample_kd_tree = preprocess_prm(env.size_x, env.size_y, rr)
# for i in xrange(len(sample_x)):
#     pl.plot([sample_x[i]], [sample_y[i]], "bo", markersize=5)
# plot_road_map(road_map, sample_x, sample_y)

# # Testing preprocess_query
# sx = x_start
# sy = y_start
# gx = x_goal
# gy = y_goal
# sample_x, sample_y = sample_points(env.size_x, env.size_y)
# for i in xrange(len(sample_x)):
#     pl.plot([sample_x[i]], [sample_y[i]], "bo", markersize=5)
# road_map, sample_kd_tree = generate_road_map(sample_x, sample_y, rr)
# full_sample_x, full_sample_y, full_road_map = preprocess_query(sx, sy, gx, gy, rr, \
#                                                                 sample_x, sample_y, road_map, sample_kd_tree)
# plot_road_map(full_road_map, full_sample_x, full_sample_y)
# print(full_road_map[-2])
# print(full_road_map[-1])

# # Testing custom query handling
sample_x, sample_y, road_map, sample_kd_tree = preprocess_prm(env.size_x, env.size_y, rr)

x_start, y_start, x_goal, y_goal = 1, 2, 8, 3
env.plot_query(x_start, y_start, x_goal, y_goal)
# Preprocess query
full_sample_x, full_sample_y, full_road_map = preprocess_query(x_start, y_start, \
                                                        x_goal, y_goal, \
                                                        rr, \
                                                        sample_x, sample_y,
                                                        road_map, sample_kd_tree)
# Handle query
rx, ry = dijkstra_planning(x_start, y_start, x_goal, y_goal, full_road_map, full_sample_x, full_sample_y)
if not rx:
    print("Cannot find a path")
else:
    print("Path is found")
    pl.plot(rx, ry, "-b")

    shorter_rx, shorter_ry = post_process(rx, ry, rr)
    pl.plot(shorter_rx, shorter_ry, "-g")

pl.show(block = True)