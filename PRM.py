import numpy as np 
import pylab as pl
import math
import copy
from scipy.spatial import cKDTree
# import matplotlib.pyplot as pl
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
pl.ion()
np.random.seed(4)
env = environment_2d.Environment(10, 6, 5)
pl.clf()
env.plot()

# Generate a random query
q = env.random_query()
if q is not None:
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)

# Definition of the key constants
MAX_EDGE_LEN = 10 # the maximum radius to find connections
N_KNN = 10 # number of edges from one sampled point
N_SAMPLE = 500 # number of sampled points
N_KNN_SPECIAL = 20 # number of edges from start point and goal points
rr = 0.01 # the minimum size to check for connections along line segment

### Definitions of all the key helper functions and class ###
def is_collision(sx, sy, gx, gy, rr):
    """
    :param sx: x coordinate of the first node
    :param sy: y coordinate of the first node
    :param gx: x coordinate of the destination node
    :param gy: y coordinates of the destination node
    :param rr: the smallest length to check
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
    dx = float(gx - sx)
    dy = float(gy - sy)
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


def generate_road_map(sample_x, sample_y, rr):
    """
    Generate the probabilistic roadmap from sample points
    :param sample_x: x coordinates of sampled points
    :param sample_y: y coordinates of sampled points
    :type sample_x, sample_y: lists
    :param rr: size to check for collision
    :type rr: float
    :return road_map: the roadmap generated
    :rtype road_map: list of lists
    :return sample_kd_tree: cKDTree containing the sample points
    :rtype sample_kd_tree: scipy.spatial.cKDTree
    """
    road_map = []
    sample_kd_tree = cKDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(N_SAMPLE), sample_x, sample_y):
        dists, indexes = sample_kd_tree.query([ix, iy], k = N_SAMPLE)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            # need to check if the dists should be used instead of is_too_long
            if not is_collision(ix, iy, nx, ny, rr) and dists[ii] <= MAX_EDGE_LEN:
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    return road_map, sample_kd_tree

def dijkstra_planning(sx, sy, gx, gy, full_road_map, full_sample_x, full_sample_y):
    """
    :param sx: starting x coordinate
    :param sy: starting y coordinate
    :param gx: goal x coordinate
    :param gy: goal y coordinate
    :type sx, sy, gy, gy: float
    :param full_road_map: road_map after adding information from query
    :type full_road_map: list of lists
    :param full_sample_x: sample_x after adding information from query
    :param full_sample_y: sample_y after adding information from query
    :type full_sample_x, full_sample_y: lists
    :return rx, ry: lists of x and y coordinates of the path, empty if no path
    :rtype rx, ry: lists
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(full_road_map) - 2] = start_node

    path_found = True

    while True:
        if not open_set:
            path_found = False
            break
        
        c_id = min(open_set, key = lambda o: open_set[o].cost)
        current = open_set[c_id]

        if c_id == len(full_road_map) - 1:
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # remove visited node from the open set
        del open_set[c_id]
        # add it to the closed set containing visited nodes
        closed_set[c_id] = current

        # expand search grid
        for i in xrange(len(full_road_map[c_id])):
            n_id = full_road_map[c_id][i]
            dx = full_sample_x[n_id] - current.x
            dy = full_sample_y[n_id] - current.y
            distance = math.hypot(dx, dy)

            node = Node(x = full_sample_x[n_id], y = full_sample_y[n_id], \
                        cost = current.cost + distance, parent_index = c_id)
            
            if n_id in closed_set:
                continue
            # Otherwise the node has not been visited
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node
   
    if path_found is False:
        return [], []

    # generate the path
    rx, ry = [goal_node.x], [goal_node.y]

    parent_index = goal_node.parent_index
    while (parent_index != -1):
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index
    
    return rx, ry

def preprocess_prm(size_x, size_y, rr):
    """
    Preprocess the given environment to generate road_map 
    :param size_x: size of the 
    :param size_x: the size of the environment in x direction
    :type size_x, size_y: float
    :param rr: size to check the collision
    :return sample_x, sample_y, road_map, sample_kd_tree
    :rtype sample_x, sample_y: lists
    :rtype road_map: list of lists
    :rtype sample_kd_tree: scipy.spatial.cKDTree
    """
    sample_x, sample_y = sample_points(size_x, size_y)
    road_map, sample_kd_tree = generate_road_map(sample_x, sample_y, rr)
    
    return sample_x, sample_y, road_map, sample_kd_tree

def preprocess_query(sx, sy, gx, gy, rr, sample_x, sample_y, road_map, sample_kd_tree):
    """
    Handle each query specifying start point and goal point
    :param sx: starting x coordinate
    :param sy: starting y coordinate
    :param gx: goal x coordinate
    :param gy: goal y coordinate
    :type sx, sy, gx, gy: float
    :return full_sample_x, full_sample_y: append startpoint and goal point
    :return full_road_map: road_map when startpoint and goal point are added
    """
    full_sample_x = copy.deepcopy(sample_x)
    full_sample_y = copy.deepcopy(sample_y)

    full_sample_x.extend([sx, gx])
    full_sample_y.extend([sy, gy])
    full_road_map = copy.deepcopy(road_map)
    for (i, ix, iy) in zip(range(N_SAMPLE, N_SAMPLE+2), \
                            full_sample_x[N_SAMPLE:], full_sample_y[N_SAMPLE:]):
        dists, indexes = sample_kd_tree.query([ix, iy], k = N_SAMPLE)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            # need to check if the dists should be used instead of is_too_long
            if not is_collision(ix, iy, nx, ny, rr) and not is_too_long(ix, iy, nx, ny, rr):
                edge_id.append(indexes[ii])
                full_road_map[indexes[ii]].append(i)

            if len(edge_id) >= N_KNN:
                break

        full_road_map.append(edge_id)
    # check between start and goal points too, since these are not in sample_kd_tree
    if not is_collision(sx, sy, gx, gy, rr) and not is_too_long(sx, sy, gx, gy, rr):
        full_road_map[-2].append(N_SAMPLE+1)
        full_road_map[-1].append(N_SAMPLE)

    return full_sample_x, full_sample_y, full_road_map

def plot_road_map(road_map, sample_x, sample_y):
    for i in xrange(len(road_map)):
        for ii in xrange(len(road_map[i])):
            ind = road_map[i][ii]

            pl.plot([sample_x[i], sample_x[ind]], \
                    [sample_y[i], sample_y[ind]], "-k") 

### Main program ###
# Preprocess the environment
sample_x, sample_y, road_map, sample_kd_tree = preprocess_prm(env.size_x, env.size_y, rr)

# Preprocess the query
full_sample_x, full_sample_y, full_road_map = preprocess_query(x_start, y_start, \
                                                                x_goal, y_goal, \
                                                                rr, \
                                                                sample_x, sample_y,
                                                                road_map, sample_kd_tree)

# plot_road_map(full_road_map, full_sample_x, full_sample_y)                                            
# Search for shortest route
rx, ry = dijkstra_planning(x_start, y_start, x_goal, y_goal, full_road_map, full_sample_x, full_sample_y)
if not rx:
    print("Cannot find a path")
else:
    print("Path is found")
    pl.plot(rx, ry, "-b")

pl.show(block=True)   


