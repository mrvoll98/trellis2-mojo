from std.time import perf_counter_ns
from trellis2_mojo.checkpoints import load_sparse_structure_flow

def main() raises:
    var t0 = perf_counter_ns()
    var m = load_sparse_structure_flow()
    var t1 = perf_counter_ns()
    print("load ss_flow:", Float64(t1 - t0) / 1e9, "s")
    _ = m^
