import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Union
import networkx as nx


def _greedy_composite(
    depths: NDArray,
    composite_length: float,
    direction: str = "forwards",
    min_length: float = 0,
) -> NDArray:
    """
    greedy composite loops over each interval creating composites of length >= the target length
    if the last sample is less or more than the target interval that is not taken into consideration
    the flag backwards simply reverses the array calculates the composites then reverses the output
    to ensure that results are ordered correctly
    Args:
        depth_from (NDArray): the from depths
        depth_to (NDArray): the to depths
        composite_length (float): the target interval depth
    Returns:
        NDArray: of intervals representing the composite intervals
    Examples:
    """
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
    # check all the intervals are > than the min_length
    # if they are smaller than the min_length then append them to
    # the previous sample
    interval_lengths = np.abs(np.diff(output))
    if np.any(interval_lengths < min_length):
        pos_short = np.where(np.diff(output) < min_length)[0]
        output = np.delete(output, pos_short)
    # if we are going backwards flip the output the right way
    if direction == "backwards":
        output = output[::-1]
    return output


def _global_composite(
    depths: NDArray,
    composite_length: float,
    min_length: float = 0,
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
    Args:
        depth_from (NDArray): the from depths
        depth_to (NDArray): the to depths
        composite_length (float): the target interval depth
        min_length (float): minimum interval length
    Returns:
        NDArray: of intervals representing the composite intervals
    Examples:
    """

    G = nx.DiGraph()
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
    interval_length: float
    # loop over each iterval starting at 0
    for i in range(maxit):
        from_depth = depths[i]
        stepstate = False
        # loop over the remaining intervals then when
        # both conditions i.e. the composite length is >= target length and
        # we have included the next composite we then move to the next interval
        for n, j in enumerate(range(i + 1, maxit)):
            to_depth = depths[j]
            interval_length = to_depth - from_depth
            weight = np.abs(composite_length - interval_length)
            if interval_length >= min_length:
                G.add_edge(i, j, weight=weight, length=to_depth - from_depth)
            if (to_depth - from_depth) > composite_length:
                stepstate = True
            if stepstate and (interval_length >= interval_length * 3):
                break

    pth = nx.shortest_path(G, source=0, target=maxit - 1, weight="weight")

    return depths[pth]


def _convert_intervals_to_from_to(intervals: NDArray) -> NDArray:
    """
    converts to intervals to from and to
    """
    composite_from: NDArray = intervals[0:-1].reshape(-1, 1)
    composite_to: NDArray = intervals[1:].reshape(-1, 1)

    return composite_from, composite_to


def composite(
    cfrom: NDArray,
    cto: NDArray,
    samplefrom: NDArray,
    sampleto: NDArray,
    array: NDArray,
    method: str = "soft",
) -> NDArray:
    """
    Simple function to composite drill hole data to a set of intervals:
    Handles compositing of data to intervals for both the hard and soft boundary conditions.
    Hard boundary data (assays, some stratigraphic contacts) does not allow for the intervals to be broken. and best effort intervals are taken.
    Soft boundary data (most geophysics, hylogger, though practically it can be ignored in most cases) does allow for the breaking of data at interval boundaries

    Args:
        cfrom (NDArray): the from depths that you would like to composite to
        cto (NDArray): the to depths that you would like to composite to
        samplefrom (NDArray): the from depth of the input array
        sampleto (NDArray): the from depth of the input array
        array (NDArray): the numpy array that you would like to have composited
    Returns:
        NDArray: with rows representing composites intervals and columns the input data
    Examples:
    """
    # reshape the from and to depths of everything to n,1
    cfrom = cfrom.reshape(-1, 1)
    cto = cto.reshape(-1, 1)
    samplefrom = samplefrom.reshape(-1, 1)
    sampleto = sampleto.reshape(-1, 1)
    # first step is to confirm that the input data is consistent
    assert cfrom.shape[0] == cto.shape[0], "Composite from to are not the same size"

    assert (
        samplefrom.shape[0] == sampleto.shape[0]
    ), "Sample from to are not the same size"

    # all we are doing now is to loop over each of the from and to intervals

    n_composites: int = cfrom.shape[0]
    n_columns: int = array.shape[1:]
    if array.ndim == 2:
        array_is_nd = False
    else:
        array_is_nd = True
    # pre-allocate the output arrays
    output: NDArray = np.zeros((n_composites, *n_columns)) * np.nan

    sample_coverage: NDArray = np.zeros((n_composites, *n_columns))

    fr: float
    to: float
    accumulated_array: NDArray
    total_weight: NDArray

    coverage: float

    sample_length = sampleto - samplefrom
    if array_is_nd:
        sample_length = sample_length.reshape(-1, 1, 1)
    # validate sample length always positive
    # if it is 0 or negative this will cause issues with the weighted sum
    # also nan values are problematic and cause low averages when included
    idx_sample_length = sample_length <= 0
    # we need to manage the dimension of this and change as dimension changes
    if array_is_nd:
        idx_sample_length = idx_sample_length.reshape(-1, 1, 1)
    # we assume that the dimension of the array is 2d at the moment
    # also if there are any nan values in any of the columns in the array
    # we will class that entire row as being nan
    idx_sample_nan = np.isnan(array)
    idx_sample_fail = idx_sample_length & idx_sample_nan
    expansion_array = np.ones((1, *n_columns))
    if method == "soft":
        cutoff = 0
    else:
        cutoff = 1
    for i in range(n_composites):
        # extract the from and to of the desired interval
        fr = cfrom[i]
        to = cto[i]
        length = to - fr
        # this is the fast and simple way
        # to calculate if a sample interval is covered by
        # a composite interval rather calculating each of the states of coverage.
        # to handle the case where we have missing samples we need to make coverage an array the
        # same size as the array of samples
        coverage = np.fmin(sampleto, to) - np.fmax(samplefrom, fr)
        if array_is_nd:
            coverage = coverage.reshape(-1, 1, 1)
        coverage = coverage * expansion_array
        # coverage will return a negative value if the sample is not inside the composite interval
        # the soft boundary case is the simplest to calculate
        # in this case we can have weights for a sample less than 1 and greater than 0
        # if we wanted the hard boundary case we would only accept weights of 1
        coverage[coverage < cutoff] = 0  # changing the 0 here
        coverage[idx_sample_nan] = 0  # manage the nan samples here
        # the matrix multiply is quite slow when applying this to a very large array
        # in the case when there are no intersections we can speed up the calculation
        # quite significantly by only carrying on the calculation if there are any samples
        # with a positive weight
        if np.any(coverage) > 0:
            # we only calculate a length weighted average
            weights = coverage / sample_length
            ## calculating the sample coverage is the sum of all the sample lengths
            total_coverage = np.clip(np.sum(coverage, 0) / length, 0, 1)

            # if the sample length is 0 or negative that will cause issues
            # use the validation index to set that weight to 0
            weights[idx_sample_fail] = 0
            total_weight = np.sum(weights, 0)
            # once we have an array of normalised weights
            # it is simple to multiple the sample array by the weights
            weight_array = weights / total_weight
            # we can speed up the calculation even more by selecting indicies
            # from the array that we are going to multiply
            idx_inside = np.any(weight_array > 0, 1)
            # then we sum the array
            # we need to use broadcasting if the array is greater than 2d

            ta = array[idx_inside].reshape(-1, *n_columns)
            tw = weight_array[idx_inside].reshape(-1, *n_columns)
            accumulated_array = np.nansum(ta * tw, axis=0, keepdims=True)
            # to manage the correct nan propagation if the  entire composite is nan values
            # we are going to replace this interval and column with nan
            # if there are gaps we will ignore them for this calculation
            idx_nan_section = np.isnan(ta).all(0)

            accumulated_array[:, idx_nan_section] = np.nan
        else:
            accumulated_array = np.nan
            total_coverage = 0
        output[i] = accumulated_array
        sample_coverage[i] = total_coverage

    return output, sample_coverage


def SoftComposite(
    samplefrom: Union[NDArray, pd.Series],
    sampleto: Union[NDArray, pd.Series],
    array: Union[NDArray, pd.DataFrame],
    interval: float = 1,
    offset: float = 0,
    drop_empty_intervals: bool = True,
    min_coverage: float = 0.1,
):
    """
    Simplifies the interface to the composite function for soft boundaries and fixed interval lengths

    Args:
        interval (float): the composite length that you would like
        samplefrom (NDArray): the from depth of the input array
        sampleto (NDArray): the from depth of the input array
        array (NDArray): the numpy array that you would like to have composited
        offset (float): offsets the start point of the intervals which are assumed to start at 0
    Returns:
        composite array (NDArray): a weighted average of the input array
    Examples:
    """
    # convert from and to if they are series to NDarray
    sfrom: NDArray
    sto: NDArray
    if isinstance(samplefrom, pd.Series):
        sfrom = samplefrom.values
    else:
        sfrom = samplefrom

    if isinstance(sampleto, pd.Series):
        sto = sampleto.values
    else:
        sto = sampleto
    # create a set of from and to depths covering the samplefrom and to depths
    min_depth: float = offset
    max_depth: float = np.max(sfrom)
    n_intervals: int = int(np.ceil(max_depth / interval))

    from_depth: NDArray = np.arange(
        min_depth, interval * n_intervals, interval
    ).reshape(-1, 1)
    to_depth: NDArray = np.arange(
        min_depth + interval, interval * (n_intervals + 1), interval
    ).reshape(-1, 1)

    sfrom = sfrom.reshape(-1, 1)
    sto = sto.reshape(-1, 1)

    # if we are dealing with a pd.DataFrame or Series then we need to strip the column headers
    isDF: bool = isinstance(array, pd.DataFrame)
    isSeries: bool = isinstance(array, pd.Series)
    clean_array: NDArray
    if isDF or isSeries:
        clean_array = array.values
    else:
        clean_array = array
    # if the array is 1d reshape it to 2d
    if clean_array.ndim == 1:
        clean_array = clean_array.reshape(-1, 1)
    # when dealing with categorical data the question of how to get the correct intervals is interesting
    # one possibility is to use dummy coding and calculate it that way
    comp_array, coverage = composite(
        from_depth,
        to_depth,
        samplefrom=sfrom,
        sampleto=sto,
        array=clean_array,
        method="soft",
    )

    depths = np.hstack([from_depth, to_depth])
    if drop_empty_intervals:
        idx_empty = np.all(coverage > min_coverage, 1)
        comp_array = comp_array[idx_empty, :]
        coverage = coverage[idx_empty, :]
        depths = depths[idx_empty, :]
    # of course you want the column headers back so we just add them back
    if isDF:
        comp_array = pd.DataFrame(comp_array, columns=array.columns)
    elif isSeries:
        comp_array = pd.Series(comp_array.ravel(), name=array.name)
    # at this point we will drop the empty intervals if that is what is wanted
    return depths, comp_array, coverage


def HardComposite(
    samplefrom: Union[NDArray, pd.Series],
    sampleto: Union[NDArray, pd.Series],
    array: Union[NDArray, pd.DataFrame],
    interval: float = 1,
    method: str = "greedy",
    direction: str = "forwards",
    drop_empty_intervals: bool = True,
    min_coverage: float = 0.1,
    min_length: float = 0,
):
    """
    Simplifies the interface to the composite function for hard boundaries which is the case
    where aggregate intervals must not cross pre-existing boundaries

    Args:
        interval (float): the composite length that you would like
        samplefrom (NDArray): the from depth of the input array
        sampleto (NDArray): the from depth of the input array
        array (NDArray): the numpy array that you would like to have composited
        min_length (float): minimum interval length
        offset (float): offsets the start point of the intervals which are assumed to start at 0
        method (str): choice of method options are 'greedy' and 'global'
        direction (str): defaults to 'forward' the direction that the greedy algorithm runs options are 'forward' and 'backward'

    Returns:
        composite array (NDArray): a weighted average of the input array composited to the target length
    Examples:
    """
    # convert from and to if they are series to NDarray
    sfrom: NDArray
    sto: NDArray
    if isinstance(samplefrom, pd.Series):
        sfrom = samplefrom.values
    else:
        sfrom = samplefrom

    if isinstance(sampleto, pd.Series):
        sto = sampleto.values
    else:
        sto = sampleto

    if min_length > interval:
        raise ValueError(
            f"minimum_length {min_length} is > interval {interval} minimum length must be <= interval length"
        )
    # if we are dealing with a pd.DataFrame or Series then we need to strip the column headers
    isDF: bool = isinstance(array, pd.DataFrame)
    isSeries: bool = isinstance(array, pd.Series)
    clean_array: NDArray
    if isDF or isSeries:
        clean_array = array.values
    else:
        clean_array = array
    # if the array is 1d reshape it to 2d
    if clean_array.ndim == 1:
        clean_array = clean_array.reshape(-1, 1)
    # when dealing with categorical data the question of how to get the correct intervals is interesting
    # one possibility is to use dummy coding and calculate it that way

    # create a set of from and to depths covering the samplefrom and to depths
    depths: NDArray = np.unique(np.concatenate([sfrom, sto]))
    if method == "greedy":
        intervals = _greedy_composite(depths, interval, direction, min_length)
    elif method == "global":
        intervals = _global_composite(depths, interval, min_length)
    else:
        raise ValueError(f'{method} must be "greedy" or "global"')
    compositefrom, compositeto = _convert_intervals_to_from_to(intervals)
    comp_array, coverage = composite(
        compositefrom, compositeto, samplefrom, sampleto, array, method="soft"
    )
    depths = np.hstack([compositefrom, compositeto])
    if drop_empty_intervals:
        idx_empty = np.all(coverage > min_coverage, 1)
        comp_array = comp_array[idx_empty, :]
        coverage = coverage[idx_empty, :]
        depths = depths[idx_empty, :]
    # of course you want the column headers back so we just add them back
    if isDF:
        comp_array = pd.DataFrame(comp_array, columns=array.columns)
    elif isSeries:
        comp_array = pd.Series(comp_array.ravel(), name=array.name)
    return depths, comp_array, coverage
