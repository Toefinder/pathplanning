import numpy as np 
import pylab as pl
import math
# import matplotlib.pyplot as pl
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

# Definition of the key constants
MAX_EDGE_LEN = 2 # the maximum radius to find connections
N_KNN = 10 # number of edges from one sampled point
N_SAMPLE = 100 # number of sampled points

def is_collision(sx, sy, gx, gy, rr):
    """
    :param sx: x coordinate of the first node
    :param sy: y coordinate of the first node
    :param gx: x coordinate of the destination node
    :param gy: y coordinates of the destination node
    :param rr: robot radius, the smallest length to check
    :type sx, sy, gx, gy, rr: float
    :return: True if the segment between the two nodes is blocked,
            False otherwise
    :rtype: boolean
    """
    x = float(sx)
    y = float(sy)
    dx = float(gx - sx)
    dy = float(gy - sy)
    distance = math.hypot(dx,dy)
    n_step = int(distance/rr) + 1 #the number of points to sample along the segment
    x_step = dx / n_step
    y_step = dy / n_step
    for i in xrange(n_step + 1):
        print(x, y)
        if env.check_collision(x, y):
            return True

        x += x_step 
        y += y_step

    return False       

def is_too_long(sx, sy, gx, gy, rr):
    """
    :param sx: x coordinate of the first node
    :param sy: y coordinate of the first node
    :param gx: x coordinate of the destination node
    :param gy: y coordinates of the destination node
    :param rr: robot radius, the smallest length to check
    :type sx, sy, gx, gy, rr: float
    :return: True if the segment between the two nodes is too long,
            False otherwise
    :rtype: boolean
    """
    distance = math.hypot(dx,dy)
    if distance > MAX_EDGE_LEN:
        return True
    else:
        return False

def sample_points(size_x, size_y):
    """
    :param size_x: the size of the environment in x direction
    :param size_y: the size of the environment in y direction
    :type size_x, size_y: float
    :return sample_x, sample_y: lists containing x and y coordinates 
    :rtype sample_x, sample_y: lists
    """
    sample_x, sample_y = [], []

    n_valid_sample = 0 # number of valid samples in Cfree
    while n_valid_sample < N_SAMPLE:
        x = np.random.rand() * size_x
        y = np.random.rand() * size_y

        if not env.check_collision(x, y):
            sample_x.append(x)
            sample_y.append(y)
            n_valid_sample += 1

    return sample_x, sample_y

class Node:
    """
    Node class to form the graph for PRM
    """      
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index
    
    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + \
                str(self.cost) + "," + str(self.parent_index)


# pl.show(block=True)   


