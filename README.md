# CARL Benchmarks

Unified 1v1 current-shared-policy self-play PPO throughput benchmarks. All runners use lookup actions, kickoff resets, GoalReward `10`, TouchReward `0.1`, goal/no-touch-10-second terminals, frameskip `8`, 3x256 policy/critic, collection `100000`, minibatch `25000`, one epoch, entropy `0.01`, and learning rate `2e-4`.

Native scales: Python `32` processes (`64` decision actors), C++ `16 x 24` games (`768` decision actors), CARL `n_sim` in `512,1024,2048,4096,8192` (`2*n_sim` decision actors).

```bash
uv sync
uv run python benchmarks/rlgymppo_python.py --iterations 5
uv run python benchmarks/jarl_carl.py --n-sim 1024 --runs 1
cmake -S benchmarks/rlgymppo_cpp -B build/rlgymppo_cpp -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/home/jack/miniconda3/envs/rlenv/bin/x86_64-conda-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/home/jack/miniconda3/envs/rlenv/bin/x86_64-conda-linux-gnu-g++ -DCMAKE_CUDA_COMPILER=/home/jack/miniconda3/envs/rlenv/bin/nvcc -DTorch_DIR=/home/jack/miniconda3/envs/rlenv/lib/python3.11/site-packages/torch/share/cmake/Torch -DPython_EXECUTABLE=/home/jack/miniconda3/envs/rlenv/bin/python -DPython_INCLUDE_DIR=/home/jack/miniconda3/envs/rlenv/include/python3.11 -DPython_LIBRARY=/home/jack/miniconda3/envs/rlenv/lib/libpython3.11.so -DPYTHON_EXECUTABLE=/home/jack/miniconda3/envs/rlenv/bin/python -DPYTHON_INCLUDE_DIR=/home/jack/miniconda3/envs/rlenv/include/python3.11 -DPYTHON_LIBRARY=/home/jack/miniconda3/envs/rlenv/lib/libpython3.11.so -DCUDA_TOOLKIT_ROOT_DIR=/tmp/opencode/cuda-rlenv -DRLGYM_COLLISION_MESHES_DIR=/home/jack/miniconda3/envs/rlenv/lib/python3.11/site-packages/rlgym/rocket_league/sim/collision_meshes
cmake --build build/rlgymppo_cpp -j 1
LD_LIBRARY_PATH=/home/jack/miniconda3/envs/rlenv/lib:/home/jack/miniconda3/envs/rlenv/targets/x86_64-linux/lib:/home/jack/miniconda3/envs/rlenv/lib/python3.11/site-packages/torch/lib ./build/rlgymppo_cpp/rlgymppo_cpp_benchmark --iterations 5 --log-file results/benchmarks.jsonl
```

All measurements append JSONL to `results/benchmarks.jsonl`; each native runner retains one warm-up iteration by default. The C++ wrapper builds against the unmodified `rlgym-ppo-cpp` submodule. CARL uses its supported custom reward callback: `score_delta` and `CarlState.car_ball_touches`, plus its built-in `no_touch_timeout_seconds`; its normal reset is its native kickoff state.
