# cudanatron

CUDA/C++ rewrite of the Catanatron game engine. Compatibility target is the
Catanatron revision pinned by the parent `catanrl` repository.

The engine is written as `__host__ __device__` functions over packed game
state so the same rules can:

- run on the CPU for parity tests and a batched MCTS coordinator;
- step thousands of environments on the GPU;
- fill contiguous neural-leaf batches instead of one-leaf Python loops.

Parity means matching Catanatron legal actions, state transitions, terminals,
flat action indices, and observations. Random streams are independent; replayed
dice, development-card draws, and robber steals must produce identical states.

## Layout

- `include/cudanatron/` packed types and APIs
- `src/` map builder, rules, action space, observations, chance, search pool, CUDA batch
- `tests/` native engine tests

## Build

```bash
cmake -S cudanatron -B cudanatron/build
cmake --build cudanatron/build
ctest --test-dir cudanatron/build --output-on-failure
```
