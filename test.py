import numpy as np

from matplotlib import pyplot as plt
from dhcomp.composite import composite
import numpy as np
import networkx as nx
import pandas as pd
from numpy.typing import NDArray

rng = np.random.default_rng(42)


def create_intervals(rnums):
    tmp = np.cumsum(rnums)
    # wrap the result
    fr = tmp[0:-1]
    to = tmp[1:]
    return fr, to


nsteps = 1000
int_1 = rng.uniform(0.9, 6, nsteps)
int_1 = rng.gamma(1, 2, nsteps)
plt.hist(int_1)
plt.show()

fr1, to1 = create_intervals(int_1)


def greedy_composite(
    depth_from: NDArray,
    depth_to: NDArray,
    composite_length: float,
    direction: str = "forwards",
) -> NDArray:
    depths: NDArray = np.unique(np.concatenate([depth_from, depth_to]))
    maxit: int = len(depths)

    if direction == "backwards":
        depths = depths[::-1]
    intervals: list[int] = []
    current_interval = 0
    intervals.append(current_interval)
    tmp_from: float
    tmp_to: float
    tmp_len: float
    i: int
    for i in range(1, maxit):
        tmp_from = depths[current_interval]
        tmp_to = depths[i]
        tmp_len = abs(tmp_to - tmp_from)
        if tmp_len >= composite_length:
            intervals.append(i)
            current_interval = i
    # check that the last interval is there and add it if missing
    if depths[intervals[-1]] != depths[maxit - 1]:
        intervals.append(maxit - 1)
    output: NDArray = depths[intervals]
    # if we are going backwards flip the output the right way
    if direction == "backwards":
        output = output[::-1]
    return output


depth_from = np.arange(0, 11, 0.1).reshape(-1, 1)
depth_to = depth_from + 0.1
step = 0.3
end = 39
depth_from, depth_to = np.arange(0, end, step), np.arange(step, end + step, step)
composite_length = 2
lengths = to1 - fr1
target_len = 2

greedy_composite(depth_from, depth_to, 2, "forwards")


def global_composite(
    depth_from: NDArray, depth_to: NDArray, composite_length: float
) -> NDArray:
    """
    compositing of drill hole intervals under the hard boundary condition
    i.e. no samples can be split between.
    this algorithm attempts to create a set of composites that are globally as
    close to the desired composite length as possible this is done by using a graph search
    algorithm provided by networkX
    additional hueristics are used to keep the run time as short as possible in particular
    if the composite interval is >= to the target interval only one additional sample is added
    this is to prevent blow out in the number of connections bwteen intervals
    """

    G = nx.Graph()
    depths: NDArray = np.unique(np.concatenate([depth_from, depth_to]))
    maxit: int = len(depths)
    # create all the nodes
    n: int
    for n, _ in enumerate(depths):
        G.add_node(n)
    # loop over all nodes
    i: int
    j: int
    n: int
    stepstate: bool
    from_depth: float
    to_depth: float
    # loop over each iterval starting at 0
    for i in range(maxit):
        from_depth = depths[i]
        stepstate = False
        # loop over the remaining intervals then when
        # both conditions i.e. the composite length is >= target length and
        # we have included the next composite we then move to the next interval
        for n, j in enumerate(range(i + 1, maxit)):
            to_depth = depths[j]
            weight = np.abs(composite_length - (to_depth - from_depth))
            G.add_edge(i, j, weight=weight, length=to_depth - from_depth)
            if (to_depth - from_depth) > composite_length:
                stepstate = True
            if stepstate and (n > 1):
                break
    pth = nx.shortest_path(G, source=0, target=maxit - 1, weight="weight")

    return depths[pth]


plt.plot(np.diff(global_composite(fr1, to1, 3.2)))
plt.show()
