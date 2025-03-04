import numpy as np
import pylab as plt
from itertools import product

arms = 3
armLength = 80 #px
start = (50,120, 170)
goal = (20,70, 40)
deltaPhi = 1

stateSpace = [] # obstacles

#combinations that are possible -> + and - and 0
actions = [a for a in product([-deltaPhi, deltaPhi, 0], repeat=arms)]
print(actions)
def a_star(start, goal, stateSpace, actions, dimension):
    open_nodes, g = {}, {}

    open_nodes[start] = 0
    g[start] = 0

    visited = {}

    while len(open_nodes) > 0:

        actual_state = min(open_nodes, key=open_nodes.get)
        del open_nodes[actual_state]

        if actual_state == goal:
            return True, visited
        
        for action in actions:
            #newState = actual_state[0] + x, actual_state[1] + y

            newState = tuple(np.array(actual_state) + np.array(action))
            # if newState[0] < 0 or newState[0] >= dimension or newState[1] < 0 or newState[1] >= dimension or stateSpace[newState[0], newState[1]] == 1:
            #     continue

            if newState not in g or g[newState] > g[actual_state] + 1:
                g[newState] = g[actual_state] + 1
                mhnt = 0
                for i in range(len(goal)):
                    mhnt += np.abs(goal[i] - newState[i])
                #print(newState, g[newState], mhnt)

                visited[newState] = g[newState] + mhnt, actual_state

                if newState not in open_nodes:
                    open_nodes[newState] = g[newState] + mhnt

                

    return False, visited


r, visited = a_star(start, goal, stateSpace, actions, armLength)
if r == True:
    # for x, y in visited.keys():
    #     stateSpace[x, y] = 2

    node = goal
    while node != start:
        f, prev_state = visited[node]
        print(prev_state)
        node = prev_state
        x, y = [0], [0]
        for i in range(len(node)):
        # x1, y1 = armLength * np.cos(np.radians(phi1)), armLength * np.sin(np.radians(phi1))
        # x2, y2 = x1 + armLength * np.cos(np.radians(phi2)), y1 + armLength * np.sin(np.radians(phi2))
            
            x1 = x[-1] + armLength * np.cos(np.radians(node[i]))
            y1 = y[-1] + armLength * np.sin(np.radians(node[i]))

            x.append(x1)
            y.append(y1)
        plt.plot(x, y)
    plt.show()
    #     row, col = prev_state
    #     stateSpace[row, col] = 3 # path

    # stateSpace[goal[0], goal[1]] = 4 # goal

    