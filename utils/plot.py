import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from environment import MazeEnv

def draw_node(state, color, radius_scale=1., dim=2, face=False):
    if dim == 2:
        facecolor = 'none'
        if face:
            facecolor = color
        circle = patches.Circle(tuple(state+1.0), radius=0.02 * radius_scale, edgecolor=color, facecolor=facecolor)
        plt.gca().add_patch(circle)

    elif dim == 3:
        a, b = MazeEnv._end_points(state)
        plt.gca().add_patch(patches.ConnectionPatch(a+1.0, b+1.0, 'data', arrowstyle="-", linewidth=2, color=color))
        plt.gca().add_patch(patches.Circle(a+1.0, radius=0.02 * radius_scale, edgecolor=color, facecolor=color))

def draw_edge(state0, state1, color, dim=2, style='-'):
    path = patches.ConnectionPatch(tuple(state0[:2]+1.0), tuple(state1[:2]+1.0), 'data', arrowstyle=style, color=color)
    plt.gca().add_patch(path)

def plot_tree(states, parents, problem, index=0, edge_classes=None):
    states = states
    environment_map = problem["map"]
    init_state = problem["init_state"]
    goal_state = problem["goal_state"]
    dim = init_state.size
    
    fig = plt.figure(figsize=(4,4))

    rect = patches.Rectangle((0.0, 0.0), 2.0, 2.0, linewidth=1, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    map_width = environment_map.shape
    d_x = 2.0 / map_width[0]
    d_y = 2.0 / map_width[1]
    for i in range(map_width[0]):
        for j in range(map_width[1]):
            if environment_map[i,j] > 0:
                rect = patches.Rectangle((d_x*i, d_y*j), d_x, d_y, linewidth=1, edgecolor='#253494', facecolor='#253494')
                plt.gca().add_patch(rect)

    for i in range(len(states)-1):
        draw_node(states[i+1], '#fdbe85', dim=dim)

        if edge_classes is None:
            draw_edge(states[i+1], states[parents[i+1]], 'green', dim=dim)
        else:
            if edge_classes[i+1] == True:
                draw_edge(states[i+1], states[parents[i+1]], 'blue', dim=dim)
            else:
                draw_edge(states[i+1], states[parents[i+1]], 'green', dim=dim)
                

    draw_node(init_state, '#e6550d', dim=dim, face=True)
    draw_node(goal_state, '#a63603', dim=dim, face=True)
    
    plt.axis([0.0, 2.0, 0.0, 2.0])
    plt.axis('off')
    plt.axis('square')

    plt.subplots_adjust(left=-0., right=1.0, top=1.0, bottom=-0.)

    plt.show()
