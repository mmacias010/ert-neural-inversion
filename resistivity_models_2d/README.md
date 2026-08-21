# ParFlow-derived resistivity slice

`resistivity2d_y2_t04368.npy` is one timestep from a ParFlow hydrologic
simulation produced by Hang Chen's group at the University of Iowa. It is
included here because `inr_benchmark.py` uses it as the `parflow` evaluation
target; the remaining 367 timesteps and the raw ParFlow output live in the
group's repository and are not redistributed here.

The slice is resampled onto the ERT mesh as a *pattern* rather than being
physically co-located, so it supplies realistic heterogeneity for benchmarking
but is not a physically registered model.
