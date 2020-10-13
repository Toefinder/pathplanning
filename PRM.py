import numpy as np 
import pylab as pl
# import matplotlib.pyplot as pl
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d
# print("helloworld")
pl.ion()
np.random.seed(4)
env = environment_2d.Environment(10, 6, 5)
pl.clf()
env.plot()
q = env.random_query()
if q is not None:
    x_start, y_start, x_goal, y_goal = q
    env.plot_query(x_start, y_start, x_goal, y_goal)
pl.show(block=True)