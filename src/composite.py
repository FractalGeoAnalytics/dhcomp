import numpy as np
import pandas as pd
from numpy.typing import NDArray


def samplestate():
    """ """


def _enclosed(
    fr: NDArray, to: NDArray, samplefrom: NDArray, sampleto: NDArray
) -> NDArray[np.bool]:
    """
    calculates the intervals that are totally covered by the from and to depths
    """
    # the first step is to do the basic check of the intervals
    idx_from: NDArray[
        np.bool
    ]  # idx of the array less than the from depth of the composite
    idx_to: NDArray[
        np.bool
    ]  # idx of the array less than the from depth of the composite
    idx_full: NDArray[np.bool]
    idx_from = samplefrom > fr
    idx_to = sampleto < to
    idx_full = idx_from & idx_to
    return idx_full


def _boundary(
    fr: NDArray, to: NDArray, samplefrom: NDArray, sampleto: NDArray
) -> NDArray[np.bool]:
    """
    returns the index of the samples intersecting the boundary
    """
    # the first step is to do the basic check of the intervals
    idx_from: NDArray[
        np.bool
    ]  # idx of the array less than the from depth of the composite
    idx_to: NDArray[
        np.bool
    ]  # idx of the array less than the from depth of the composite
    idx_partial: NDArray[np.bool]
    idx_from = samplefrom <= to
    idx_to = sampleto >= fr
    idx_partial = idx_from & idx_to
    return idx_partial


def _contact(idx_partial, idx_enclosed):
    return np.logical_xor(idx_partial, idx_enclosed)

def _sample_weight():

def composite(
    cfrom: NDArray, cto: NDArray, samplefrom: NDArray, sampleto: NDArray, array: NDArray
):
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

    Examples:
    """
    # first step is to confirm that the input data is consistent
    assert cfrom.shape[0] == cto.shape[0], "Composite from to are not the same size"

    assert (
        samplefrom.shape[0] == sampleto.shape[0]
    ), "Sample from to are not the same size"

    # all we are doing now is to loop over each of the from and to intervals

    n_composites: int = cfrom.shape[0]
    n_columns: int = array.shape[1]
    output: NDArray = np.zeros((n_composites, n_columns))*np.nan
    fr: float
    to: float
    accumulated_array:NDArray
    total_weight:NDArray
    idx_enclosed:NDArray[np.bool]
    idx_boundary:NDArray[np.bool]
    idx_contact:NDArray[np.bool]
    coverage:float

    sample_length = sampleto-samplefrom
    for i in range(n_composites):
        fr = cfrom[i]
        to = cto[i]
        idx_enclosed = _enclosed(fr, to, samplefrom, sampleto)
        idx_boundary = _boundary(fr, to, samplefrom, sampleto)
        idx_contact = _contact(idx_boundary, idx_enclosed)
        if np.any(idx_contact):
            # calculate the weight
            # https://blogs.sas.com/content/sgf/2022/01/13/calculating-the-overlap-of-date-time-intervals/
            weights = idx_contact.astype(float)
            for j in np.where(idx_contact)[0]:
                coverage =  max(0,min(samplefrom[j],fr)-min(sampleto[j],to)+1)
                temp_weight = sample_length[j]/coverage
                weights[j] =  temp_weight
        else:
            weights = idx_boundary

        total_weight = np.sum(weights)
        
        weight_array = weights.reshape(-1,1)/total_weight
        accumulated_array = np.sum(array*weight_array,0)
        output[i,:] = accumulated_array/total_weight

    return output
