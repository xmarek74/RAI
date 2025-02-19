### EXERCISE 1 - IMPLEMENTING DFS USING RECURSION ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def dfs(state, goal, depth, space, visited, actions, dim):
    if (state == goal):
        return ("Solution has been found")
    visited.add(state)
    if (depth == 0):
        return ("Depth limit has been reached")
    x, y = state
    for m, n in actions:
        newState = x + m, y + n
        if (newState[0] and newState[1] > 0) and (newState[0] and newState[1] < dim) and newState not in visited:
            return dfs(newState, goal, depth-1, space, visited, actions, dim)    
def runDfs(start, goal, maxDepth, dimensions, visited):
    #create space
    space = np.zeros([dimensions, dimensions])
    #actions representing movement through matrix
    actions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    print(dfs(start, goal, maxDepth, space, visited, actions, dimensions))
    #visualize the path and set up plot
    for i, j in visited:
        space[i, j] = 3
    space[start[0], start[1]] = 1
    space[goal[0], goal[1]] = 2

    #create color map
    cmap = mcolors.ListedColormap(['white', 'green', 'red', 'orange'])  
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(space, cmap=cmap, norm=norm)
    plt.show()

runDfs((10,10), (10, 30), 500, 50, visited=set())

### EXERCISE 2 - IMPLEMENTING A* ###
