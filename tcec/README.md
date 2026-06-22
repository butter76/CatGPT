# CatGPT - TCEC submission

CatGPT-LKS is a GPU / neural-network chess engine (JAX-trained policy+value net
exported to ONNX, served through TensorRT) with a libfork-based parallel search.

- Source: https://github.com/butter76/CatGPT.git
- Version: the engine self-reports its build commit. Run `./catgpt --version`
  (or send `uci` and read `id name`), e.g. `CatGPT-LKS commit <sha> (<date>)`.
- TODO: tournament / season + link
- TODO: time control (e.g. 90 min + 10 s/move)
- TODO: hardware (CPU model, cores, RAM, GPU model, SSD)
- TODO: seed Elo (set `rating` in `engines.json`)

## Build / update

The build is driven by `update.sh` (clones the repo, fetches the `S4.onnx`
network, builds GCC 14 + TensorRT + the engine via `scripts/build.sh`, packs
the per-bucket `S4.network` engine bundle, and finishes by running
`trt_benchmark` on it). `trt_benchmark` is the build self-test in lieu of a UCI
`bench` subcommand (this engine has no `bench` command); it prints per-batch
throughput so operators can confirm the binary and GPU are healthy.

Required on the host (cannot be installed without root): an NVIDIA driver and a
CUDA 12.x toolkit. Everything else is built into the working directory. After a
successful run the working directory contains `catgpt` (wrapper), `catgpt.real`
(ELF), and `S4.network`.

## Configuration: CLI flags and env vars (no UCI options)

This engine exposes **no settable UCI options**; `setoption` is ignored.
Configure it instead with CLI flags (preferred for TCEC) or `LKS_*` environment
variables. Precedence is **CLI flag > env var > built-in default**. See
`./catgpt --help` for the full list. Common knobs:

| CLI flag | Env var | Default | Meaning |
|---|---|---|---|
| `--network PATH` (or positional) | `CATGPT_TRT_ENGINE` | `./S4.network` | TensorRT engine bundle |
| `--syzygy-path PATH` | `LKS_SYZYGY_PATH` / `SYZYGY_HOME` | disabled | Syzygy tablebase directory |
| `--workers-per-gpu N` | `LKS_WORKERS_PER_GPU` | 2 | search workers per GPU |
| `--coros-per-worker N` | `LKS_COROS_PER_WORKER` | 256 | coroutines per worker |
| `--max-batch N` | `LKS_MAX_BATCH_SIZE` | 112 | max eval batch size |
| `--max-evals N` | `LKS_LIFETIME_MAX_EVALS` | `1<<27` | lifetime eval / arena cap |

Time-management knobs are exposed as `--time-*` flags (mirroring `LKS_TIME_*`).

- **Pondering**: the engine never ponders and exposes no Ponder option, so the
  TCEC "permanent brain off" rule is satisfied with no configuration.
- **Tablebases**: TCEC mounts Syzygy on the server; the build does not download
  them. The path is passed via `--syzygy-path` in `engines.json` (`/home/syzygy`).

## engines.json

`engines.json` is a cutechess-cli config launched from
`/home/tcec/Engines/CatGPT/`. The Syzygy path is part of `command` (it is a CLI
flag, not a UCI option), and `options` is intentionally empty. Append extra
`--flags` to `command` to tune compute once the TCEC hardware is known, e.g.:

```
"command": "./catgpt S4.network --syzygy-path /home/syzygy --workers-per-gpu 2 --max-batch 112"
```
