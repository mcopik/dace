# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""Simple program showing the DaCe Python interface via scalar multiplication and vector addition."""

import argparse
import dace
import numpy as np

from dace.transformation.interstate import WormholeTransformSDFG
from dace.transformation.dataflow import MapTiling
from dace.transformation.dataflow import InLocalStorage, OutLocalStorage

# Define a symbol so that the vectors could have arbitrary sizes and compile the code once
# (this step is not necessary for arrays with known sizes)
N = dace.symbol("N")


# Define the data-centric program with type hints
# (without this step, Just-In-Time compilation is triggered every call)
# @dace.program(auto_optimize=True, device=dace.dtypes.DeviceType.GPU)
@dace.program(
    auto_optimize=False,
    device=dace.dtypes.DeviceType.WORMHOLE,
    regenerate_code=True,
    recompile=True,
)
def axpy(x: dace.float64[N], y: dace.float64[N]):
    result = dace.ndarray([N], dace.float64)
    for i in dace.map[0:N] @ dace.ScheduleType.Wormhole_Kernel:
        result[i] = np.add(x[i], y[i])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=1024)
    args = parser.parse_args()

    # Initialize arrays
    a = np.random.rand()
    x = np.random.rand(args.N)
    y = np.random.rand(args.N)

    # Call the p rogram (the value of N is inferred by dace automatically)
    sdfg = axpy.to_sdfg()
    # sdfg.save("wormhole0.sdfg")
    sdfg.apply_transformations(WormholeTransformSDFG)
    # sdfg.save("wormhole1.sdfg")
    sdfg.apply_transformations(MapTiling, {"tile_sizes": (128,), "divides_evenly": True})
    # sdfg.save("wormhole2.sdfg")
    sdfg.apply_transformations_repeated([InLocalStorage, OutLocalStorage])
    sdfg.save("wormhole3.sdfg")
    sdfg.compile()
    z = sdfg(x, y)

    # Check result
    # expected = x + y
    # print("Difference:", np.linalg.norm(z - expected))
