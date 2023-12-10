import numpy as np

def over_the_top_neighbors(i, n, xdim=120):
    return (np.linspace(0, xdim-1, n+2, dtype=int)[1:-1] + i ) % (xdim)

for i in range(120):
    print(over_the_top_neighbors(i, 3, xdim=120))

