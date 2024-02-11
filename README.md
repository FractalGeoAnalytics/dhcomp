# dhcomp

## Rationale
There does not seem to be any permissively licenced drill hole compositing software in python.
dhcomp is a MIT licenced open source one function utility that (currently) composites geophysical data to a set of intervals.


## Installation
Installation
```pip install dhcomp```

## Usage 
```python
from matplotlib import pyplot as plt
from dhcomp.composite import composite
import numpy as np
rng = np.random.default_rng(42)
nsteps = 100
int_1 = rng.gamma(1,4,nsteps)
int_2 = rng.gamma(1,2,nsteps)

def create_intervals(rnums):

    tmp = np.cumsum(rnums)
    # wrap the result
    fr = tmp[0:-1]
    to = tmp[1:]
    return fr, to

fr1, to1 = create_intervals(int_1)
fr2, to2 = create_intervals(int_2)
# use nsteps-1 because we loose a step
v1 = np.cumsum(rng.standard_normal(nsteps-1))
v2 = np.cumsum(rng.standard_normal(nsteps-1))
c2,_ = composite(fr1, to1, fr2, to2, v2.reshape(-1,1))

plt.plot(fr1,v1,'.-',label='process 1')
plt.plot(fr2,v2,'.-',label='process 2')
plt.plot(fr1,c2,'.-',label='process 2 values composited\nto process 1 intervals')
plt.legend()
plt.xlabel('steps')
plt.ylabel('value')
plt.title('resampled irregular time series')
plt.show()
```

## Usage

https://www.fractalgeoanalytics.com/articles/2023-01-13-compositing-drill-hole-intervals/
