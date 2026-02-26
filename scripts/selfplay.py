#!/usr/bin/env python3
"""CatGPT Selfplay Tournament with W&B logging.

Launches the C++ batched selfplay binary (catgpt_selfplay) and streams
real-time metrics to Weights & Biases.

The C++ binary runs the actual tournament; this script handles
configuration, W&B initialization, and metric ingestion via JSON-lines
output from the binary's stdout (enabled by --json-metrics).

Usage:
    # Run with defaults (prompts for run name)
    uv run python scripts/selfplay.py

    # Override search params
    uv run python scripts/selfplay.py evals=800 cpuct=2.0

    # Override per-engine
    uv run python scripts/selfplay.py challenger_evals=800 baseline_evals=400

    # Disable W&B
    uv run python scripts/selfplay.py wandb.enabled=false
"""

import difflib
import json
import subprocess
import sys
import threading
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def prompt_run_info() -> tuple[str, str]:
    """Prompt user for a run name and description."""
    print("\n" + "=" * 50)
    print("CatGPT Selfplay Tournament")
    print("=" * 50)
    run_name = input("Enter run name (for W&B / PGN): ").strip()
    if not run_name:
        raise ValueError("Run name cannot be empty")
    run_name = run_name.replace(" ", "_")
    run_name = "".join(c for c in run_name if c.isalnum() or c in "_-")
    print(f"Run name: {run_name}")
    print("-" * 50)
    run_description = input("Describe this run (for W&B notes): ").strip()
    print("=" * 50 + "\n")
    return run_name, run_description


def build_command(cfg: DictConfig, project_root: Path, pgn_path: Path) -> list[str]:
    """Build the catgpt_selfplay command line from config."""
    binary = project_root / cfg.cpp_build_dir / "catgpt_selfplay"
    if not binary.exists():
        raise FileNotFoundError(
            f"Selfplay binary not found: {binary}\n"
            "Build with: cd cpp/build && cmake .. && make catgpt_selfplay -j$(nproc)"
        )

    trt_engine = project_root / cfg.trt_engine
    if not trt_engine.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {trt_engine}")

    cmd = [str(binary), str(trt_engine)]

    # Tournament settings
    if cfg.pairs:
        cmd += ["--pairs", str(cfg.pairs)]
    cmd += ["--concurrent", str(cfg.concurrent)]
    cmd += ["--threads", str(cfg.threads)]
    cmd += ["--batch", str(cfg.batch)]

    # Search settings (shared)
    cmd += ["--evals", str(cfg.evals)]
    cmd += ["--cpuct", str(cfg.cpuct)]

    # Per-engine overrides
    if cfg.baseline_evals is not None:
        cmd += ["--baseline-evals", str(cfg.baseline_evals)]
    if cfg.challenger_evals is not None:
        cmd += ["--challenger-evals", str(cfg.challenger_evals)]
    if cfg.baseline_cpuct is not None:
        cmd += ["--baseline-cpuct", str(cfg.baseline_cpuct)]
    if cfg.challenger_cpuct is not None:
        cmd += ["--challenger-cpuct", str(cfg.challenger_cpuct)]

    # Labels
    cmd += ["--baseline-name", cfg.baseline_name]
    cmd += ["--challenger-name", cfg.challenger_name]

    # Openings
    openings_path = project_root / cfg.openings
    if openings_path.exists():
        cmd += ["--openings", str(openings_path)]
    else:
        logger.warning(f"Openings file not found: {openings_path}, using startpos")

    # Syzygy tablebase path (explicit config or $SYZYGY_HOME handled by C++ binary)
    if cfg.syzygy_path is not None:
        syzygy = project_root / cfg.syzygy_path if not Path(cfg.syzygy_path).is_absolute() else Path(cfg.syzygy_path)
        if syzygy.exists():
            cmd += ["--syzygy", str(syzygy)]
        else:
            logger.warning(f"Syzygy path not found: {syzygy}, falling back to $SYZYGY_HOME")

    # PGN output
    cmd += ["--pgn", str(pgn_path)]

    # JSON metrics for this wrapper to consume
    cmd += ["--json-metrics"]

    return cmd


@hydra.main(version_base=None, config_path="../configs", config_name="selfplay")
def main(cfg: DictConfig) -> None:
    """Main selfplay entry point."""
    run_name, run_description = prompt_run_info()

    # Hydra changes cwd, so resolve project root
    project_root = Path(hydra.utils.get_original_cwd())

    setup_logging(cfg.verbose)
    logger.info("Starting CatGPT Selfplay Tournament")
    logger.info(f"Run name: {run_name}")
    if run_description:
        logger.info(f"Description: {run_description}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # PGN path
    if cfg.pgn_path is not None:
        pgn_path = project_root / cfg.pgn_path
    else:
        pgn_path = project_root / "outputs" / f"selfplay_{run_name}.pgn"
    pgn_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb as wb

            wandb_config = OmegaConf.to_container(cfg, resolve=True)
            # Capture git metadata before init so it lands in the config
            git_commit = None
            git_diff_text = None
            try:
                git_commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(project_root),
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
                git_diff_text = (
                    subprocess.check_output(
                        ["git", "diff", "HEAD"],
                        cwd=str(project_root),
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                )
                wandb_config["git_commit"] = git_commit
                if git_diff_text:
                    wandb_config["git_dirty"] = True
                else:
                    wandb_config["git_dirty"] = False
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Could not read git info — not a git repo or git not installed")

            wandb_run = wb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                notes=run_description or None,
                tags=list(cfg.wandb.tags),
                config=wandb_config,
            )
            logger.info(f"W&B initialized: {wb.run.url}")

            # Upload git diff as a patch file
            if git_diff_text:
                git_diff_path = project_root / "outputs" / f"git_diff_{run_name}.patch"
                git_diff_path.parent.mkdir(parents=True, exist_ok=True)
                git_diff_path.write_text(git_diff_text)
                wb.save(str(git_diff_path), base_path=str(project_root), policy="now")
                logger.info(f"W&B tracking git diff ({len(git_diff_text)} bytes, commit {git_commit})")
            elif git_commit:
                logger.info(f"Clean working tree at commit {git_commit}")

            # Track search source files so each run records exactly what code was used
            baseline_path = project_root / "cpp" / "src" / "selfplay" / "coroutine_search.hpp"
            challenger_path = project_root / "cpp" / "src" / "selfplay" / "challenger_search.hpp"

            for f in [baseline_path, challenger_path]:
                if f.exists():
                    wb.save(str(f), base_path=str(project_root), policy="now")
                    logger.info(f"W&B tracking: {f.relative_to(project_root)}")

            # Generate and upload a unified diff between baseline and challenger
            if baseline_path.exists() and challenger_path.exists():
                baseline_lines = baseline_path.read_text().splitlines(keepends=True)
                challenger_lines = challenger_path.read_text().splitlines(keepends=True)
                diff = difflib.unified_diff(
                    baseline_lines,
                    challenger_lines,
                    fromfile="coroutine_search.hpp (baseline)",
                    tofile="challenger_search.hpp (challenger)",
                )
                diff_text = "".join(diff)
                if diff_text:
                    diff_path = project_root / "outputs" / f"challenger_diff_{run_name}.patch"
                    diff_path.parent.mkdir(parents=True, exist_ok=True)
                    diff_path.write_text(diff_text)
                    wb.save(str(diff_path), base_path=str(project_root), policy="now")
                    logger.info(f"W&B tracking diff: {diff_path.relative_to(project_root)}")
                else:
                    logger.info("Baseline and challenger are identical — no diff to upload")

        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    # Build command
    try:
        cmd = build_command(cfg, project_root, pgn_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)

    logger.info(f"Command: {' '.join(cmd)}")

    # Launch C++ binary.
    # stdout = JSON metrics (parsed by this script)
    # stderr = human-readable logs (piped through Python's stderr so wandb captures them)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line-buffered
    )

    # Forward C++ stderr to Python's stderr in a background thread.
    # This ensures wandb's automatic console log capture sees the output.
    def _forward_stderr() -> None:
        for line in process.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()

    stderr_thread = threading.Thread(target=_forward_stderr, daemon=True)
    stderr_thread.start()

    games_logged = 0
    summary_data = None

    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Not JSON — skip (shouldn't happen since stdout is only JSON)
                logger.debug(f"Non-JSON stdout: {line}")
                continue

            msg_type = data.get("type")

            if msg_type == "game":
                games_logged += 1
                total = data.get("games", 0)
                wins = data.get("challenger_wins", 0)
                draws = data.get("draws", 0)
                losses = data.get("challenger_losses", 0)

                if wandb_run:
                    import wandb as wb

                    wb.log(
                        {
                            "games": total,
                            "challenger_wins": wins,
                            "draws": draws,
                            "challenger_losses": losses,
                            "elo_estimate": data.get("elo", 0),
                            "win_rate": wins / total if total > 0 else 0,
                            "draw_rate": draws / total if total > 0 else 0,
                            "loss_rate": losses / total if total > 0 else 0,
                            "game_moves": data.get("game_moves", 0),
                            "game_gpu_evals": data.get("game_gpu_evals", 0),
                            "avg_moves_per_game": data.get("avg_moves", 0),
                            "avg_gpu_evals_per_game": data.get("avg_gpu_evals", 0),
                            "games_per_sec": data.get("games_per_sec", 0),
                        }
                    )

            elif msg_type == "summary":
                summary_data = data
                logger.info(
                    f"Tournament complete: {data.get('games', 0)} games, "
                    f"Elo: {data.get('elo', 0):+.1f}, "
                    f"{data.get('games_per_sec', 0):.1f} games/sec"
                )

    except KeyboardInterrupt:
        logger.warning("Interrupted — terminating C++ process")
        process.terminate()
    finally:
        process.wait()
        stderr_thread.join(timeout=5)

    # Log summary to W&B
    if wandb_run and summary_data:
        import wandb as wb

        wb.run.summary.update(
            {
                "final/games": summary_data.get("games", 0),
                "final/challenger_wins": summary_data.get("challenger_wins", 0),
                "final/draws": summary_data.get("draws", 0),
                "final/challenger_losses": summary_data.get("challenger_losses", 0),
                "final/elo": summary_data.get("elo", 0),
                "final/avg_moves": summary_data.get("avg_moves", 0),
                "final/avg_gpu_evals": summary_data.get("avg_gpu_evals", 0),
                "final/total_secs": summary_data.get("total_secs", 0),
                "final/games_per_sec": summary_data.get("games_per_sec", 0),
                "final/total_gpu_evals": summary_data.get("total_gpu_evals", 0),
                "final/gpu_evals_per_sec": summary_data.get("gpu_evals_per_sec", 0),
            }
        )

    if wandb_run:
        # Log PGN as artifact if it exists
        if pgn_path.exists():
            try:
                import wandb as wb

                artifact = wb.Artifact(f"selfplay-pgn-{run_name}", type="pgn")
                artifact.add_file(str(pgn_path))
                wandb_run.log_artifact(artifact)
                logger.info(f"Uploaded PGN artifact: {pgn_path}")
            except Exception as e:
                logger.warning(f"Failed to upload PGN artifact: {e}")

        wandb_run.finish()

    exit_code = process.returncode
    if exit_code != 0:
        logger.error(f"C++ process exited with code {exit_code}")
        raise SystemExit(exit_code)

    logger.info(f"Done. PGN saved to: {pgn_path}")


if __name__ == "__main__":
    main()
