import pyperf
from world_engine import WorldEngine, CtrlInput

import warnings
warnings.filterwarnings("ignore")


# TODO
# - benchmark encode img
# - benchmark encode prompt
# - isolated benchmark: decode image time

# TODO:
"""
world_engine version: <version / commit>

Hardware:
- details
- details

Benchmarks:
- benchmark
- benchmark
"""


def gen_n_frames(engine, n_frames):
    for _ in range(n_frames):
        engine.gen_frame()


def run_benchmark(model_uri: str = "OpenWorldLabs/CoDCtl-Causal", device: str = "cuda") -> None:
    engine = WorldEngine(model_uri, device=device, model_config_overrides={"n_frames": 512})

    # Warmup torch compilation
    for _ in range(3):
        engine.gen_frame()

    runner = pyperf.Runner(processes=1)

    for n_frames in [1, 4, 16, 64]:
        runner.timeit(
            f"AR Rollout n_frames={n_frames}",
            stmt="gen_n_frames(engine, n_frames)",
            teardown="engine.reset()",
            globals={"engine": engine, "gen_n_frames": gen_n_frames, "n_frames": n_frames},
        )


if __name__ == "__main__":
    run_benchmark()
