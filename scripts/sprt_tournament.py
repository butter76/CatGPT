#!/usr/bin/env python3
"""SPRT Tournament runner for C++ chess engines.

Run engine-vs-engine matches using Sequential Probability Ratio Test (SPRT)
to determine if one engine configuration is statistically stronger than another.

Usage:
    # Run with defaults (prompts for run name)
    uv run python scripts/sprt_tournament.py

    # Compare different MCTS node counts
    uv run python scripts/sprt_tournament.py \\
        engine_a.type=mcts engine_a.mcts.num_simulations=400 \\
        engine_b.type=mcts engine_b.mcts.num_simulations=800

    # Compare different engine types
    uv run python scripts/sprt_tournament.py \\
        engine_a.type=fractional_mcts \\
        engine_b.type=mcts

    # Use different models
    uv run python scripts/sprt_tournament.py \\
        engine_a.trt_engine=new_model.trt \\
        engine_b.trt_engine=baseline.trt

    # Tighter SPRT bounds (more games, more precision)
    uv run python scripts/sprt_tournament.py sprt.elo1=5

    # Disable W&B logging
    uv run python scripts/sprt_tournament.py wandb.enabled=false
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from catgpt.cpp.uci_engine import (
    FractionalMCTSEngine,
    MCTSEngine,
    PolicyEngine,
    UCIEngine,
    ValueEngine,
)
from catgpt.tournament.game_runner import (
    GameConfig,
    GameResult,
    GameRunner,
    GameTermination,
)
from catgpt.tournament.openings import load_openings
from catgpt.tournament.sprt import SPRTCalculator, SPRTResult, SPRTStatus

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    import sys

    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def prompt_run_name() -> str:
    """Prompt user for a run name to organize W&B runs."""
    print("\n" + "=" * 50)
    print("CatGPT SPRT Tournament")
    print("=" * 50)
    run_name = input("Enter run name (for W&B): ").strip()
    if not run_name:
        raise ValueError("Run name cannot be empty")
    run_name = run_name.replace(" ", "_")
    run_name = "".join(c for c in run_name if c.isalnum() or c in "_-")
    print(f"W&B run name: {run_name}")
    print("=" * 50 + "\n")
    return run_name


def create_engine(
    engine_cfg: DictConfig, cpp_build_dir: Path, project_root: Path
) -> UCIEngine:
    """Create a UCI engine from configuration.

    Args:
        engine_cfg: Engine configuration section.
        cpp_build_dir: Path to C++ build directory.
        project_root: Project root for resolving paths.

    Returns:
        Configured UCI engine instance.
    """
    trt_engine = project_root / engine_cfg.trt_engine
    timeout = engine_cfg.timeout

    binary_map = {
        "mcts": "catgpt_mcts",
        "fractional_mcts": "catgpt_fractional_mcts",
        "value": "catgpt_value",
        "policy": "catgpt_policy",
    }

    engine_type = engine_cfg.type
    if engine_type not in binary_map:
        raise ValueError(f"Unknown engine type: {engine_type}")

    # Use custom binary_path if provided, otherwise auto-derive from type
    if engine_cfg.get("binary_path") is not None:
        custom_path = Path(engine_cfg.binary_path)
        # Resolve relative paths against project root
        if custom_path.is_absolute():
            binary_path = custom_path
        else:
            binary_path = project_root / custom_path
    else:
        binary_path = cpp_build_dir / binary_map[engine_type]

    if not binary_path.exists():
        raise FileNotFoundError(
            f"Engine binary not found: {binary_path}\n"
            f"Build with: cd cpp/build && cmake .. && make {binary_map[engine_type]} -j$(nproc)"
        )

    if not trt_engine.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {trt_engine}")

    if engine_type == "mcts":
        num_simulations = engine_cfg.mcts.num_simulations
        logger.info(f"Creating MCTS engine ({num_simulations} simulations)")
        return MCTSEngine(
            binary_path, trt_engine, num_simulations=num_simulations, timeout=timeout
        )
    elif engine_type == "fractional_mcts":
        min_evals = engine_cfg.fractional_mcts.min_total_evals
        logger.info(f"Creating Fractional MCTS engine ({min_evals} min evals)")
        return FractionalMCTSEngine(
            binary_path, trt_engine, min_total_evals=min_evals, timeout=timeout
        )
    elif engine_type == "value":
        logger.info("Creating Value engine (1-ply)")
        return ValueEngine(binary_path, trt_engine, timeout=timeout)
    else:  # policy
        logger.info("Creating Policy engine")
        return PolicyEngine(binary_path, trt_engine, timeout=timeout)


def create_status_table(
    sprt_result: SPRTResult,
    engine_a_name: str,
    engine_b_name: str,
    current_opening: int,
    total_openings: int,
) -> Table:
    """Create a rich table showing current SPRT status."""
    table = Table(title="SPRT Tournament Status", show_header=True)

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Game stats
    table.add_row("Games Played", str(sprt_result.games))
    table.add_row(
        f"{engine_a_name} Wins", f"{sprt_result.wins} ({sprt_result.win_rate:.1%})"
    )
    table.add_row(
        f"{engine_b_name} Wins", f"{sprt_result.losses} ({sprt_result.loss_rate:.1%})"
    )
    table.add_row("Draws", f"{sprt_result.draws} ({sprt_result.draw_rate:.1%})")
    table.add_row("", "")

    # SPRT stats
    table.add_row("Score", f"{sprt_result.score:.3f}")
    table.add_row(
        "Elo Estimate", f"{sprt_result.elo_estimate:+.1f} ± {sprt_result.elo_error:.1f}"
    )
    table.add_row("", "")

    # LLR
    llr_color = (
        "green"
        if sprt_result.llr > 0
        else "red" if sprt_result.llr < 0 else "yellow"
    )
    table.add_row(
        "LLR",
        f"[{llr_color}]{sprt_result.llr:.3f}[/{llr_color}] "
        f"([{sprt_result.lower_bound:.3f}, {sprt_result.upper_bound:.3f}])",
    )

    # Status
    status_color = {
        SPRTStatus.CONTINUE: "yellow",
        SPRTStatus.H1_ACCEPTED: "green",
        SPRTStatus.H0_ACCEPTED: "red",
    }[sprt_result.status]
    table.add_row("Status", f"[{status_color}]{sprt_result.status.value}[/{status_color}]")

    # Progress
    table.add_row("", "")
    table.add_row("Opening", f"{current_opening + 1} / {total_openings}")

    return table


class PGNWriter:
    """Incremental PGN writer that flushes games as they complete."""

    def __init__(
        self,
        path: Path,
        engine_a_name: str,
        engine_b_name: str,
        event: str = "SPRT Tournament",
    ):
        self.path = path
        self.engine_a_name = engine_a_name
        self.engine_b_name = engine_b_name
        self.event = event
        self.game_count = 0
        self._file = None

    def open(self) -> None:
        """Open the PGN file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w")
        logger.info(f"PGN output: {self.path}")

    def write_game(self, game: GameResult) -> None:
        """Write a single game to the PGN file and flush."""
        if self._file is None:
            return

        self.game_count += 1
        pgn = game.to_pgn(
            engine_a_name=self.engine_a_name,
            engine_b_name=self.engine_b_name,
            event=self.event,
            round_num=self.game_count,
        )
        self._file.write(pgn)
        self._file.write("\n")
        self._file.flush()  # Flush immediately so games are saved as they finish

    def write_games(self, games: list[GameResult]) -> None:
        """Write multiple games."""
        for game in games:
            self.write_game(game)

    def close(self) -> None:
        """Close the PGN file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(f"Saved {self.game_count} games to {self.path}")


@hydra.main(version_base=None, config_path="../configs", config_name="sprt_tournament")
def main(cfg: DictConfig) -> None:
    """Main SPRT tournament entry point."""
    # Prompt for run name
    run_name = prompt_run_name()

    # Get project root (Hydra changes cwd)
    project_root = Path(hydra.utils.get_original_cwd())

    # Setup logging
    setup_logging(cfg.verbose)
    logger.info("Starting SPRT Tournament")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize W&B
    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb as wb

            wandb_run = wb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                tags=list(cfg.wandb.tags),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info(f"W&B initialized: {wb.run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    # Load opening book
    openings_path = project_root / cfg.openings.path
    if openings_path.exists():
        openings = load_openings(
            openings_path,
            shuffle=cfg.openings.shuffle,
            seed=cfg.openings.seed,
        )
    else:
        raise FileNotFoundError(f"Opening book not found: {openings_path}")

    logger.info(f"Loaded {len(openings)} opening positions")

    # Create engines
    cpp_build_dir = project_root / cfg.cpp_build_dir

    logger.info("Creating Engine A...")
    engine_a = create_engine(cfg.engine_a, cpp_build_dir, project_root)
    engine_a_name = engine_a.name

    logger.info("Creating Engine B...")
    engine_b = create_engine(cfg.engine_b, cpp_build_dir, project_root)
    engine_b_name = engine_b.name

    logger.info(f"Engine A: {engine_a_name}")
    logger.info(f"Engine B: {engine_b_name}")

    # Create SPRT calculator
    sprt = SPRTCalculator(
        elo0=cfg.sprt.elo0,
        elo1=cfg.sprt.elo1,
        alpha=cfg.sprt.alpha,
        beta=cfg.sprt.beta,
    )
    logger.info(
        f"SPRT bounds: elo0={cfg.sprt.elo0}, elo1={cfg.sprt.elo1}, "
        f"alpha={cfg.sprt.alpha}, beta={cfg.sprt.beta}"
    )
    logger.info(
        f"LLR bounds: [{sprt.lower_bound:.3f}, {sprt.upper_bound:.3f}]"
    )
    logger.info(f"Estimated games: ~{sprt.games_estimate()}")

    # Create game runner
    # Resolve Syzygy path if enabled
    syzygy_path = None
    if cfg.game.syzygy.enabled:
        syzygy_path_cfg = Path(cfg.game.syzygy.path)
        # Use absolute path as-is, otherwise resolve relative to project root
        if syzygy_path_cfg.is_absolute():
            syzygy_path = str(syzygy_path_cfg)
        else:
            syzygy_path = str(project_root / syzygy_path_cfg)
        logger.info(f"Syzygy tablebases enabled: {syzygy_path}")

    game_config = GameConfig(
        adjudicate_draw_moves=cfg.game.adjudicate_draw_moves,
        adjudicate_draw_score=cfg.game.adjudicate_draw_score,
        adjudicate_draw_count=cfg.game.adjudicate_draw_count,
        adjudicate_resign_score=cfg.game.adjudicate_resign_score,
        adjudicate_resign_count=cfg.game.adjudicate_resign_count,
        syzygy_enabled=cfg.game.syzygy.enabled,
        syzygy_path=syzygy_path,
        syzygy_adjudicate_draw=cfg.game.syzygy.adjudicate_draw,
        syzygy_adjudicate_win=cfg.game.syzygy.adjudicate_win,
    )
    runner = GameRunner(game_config)

    # Track results
    wins, losses, draws = 0, 0, 0
    syzygy_adjudications = 0
    all_games: list[GameResult] = []

    # Setup PGN writer (writes games incrementally as they finish)
    pgn_writer: PGNWriter | None = None
    if cfg.output.save_pgn:
        if cfg.output.pgn_path:
            pgn_path = project_root / cfg.output.pgn_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pgn_path = project_root / f"outputs/sprt_{run_name}_{timestamp}.pgn"

        pgn_writer = PGNWriter(
            pgn_path,
            engine_a_name=engine_a_name,
            engine_b_name=engine_b_name,
            event=f"SPRT: {run_name}",
        )
        pgn_writer.open()

    # Main tournament loop
    max_rounds = cfg.sprt.max_games // 2  # 2 games per round (color swap)
    opening_idx = 0

    try:
        console.print(f"\n[bold]Starting SPRT: {engine_a_name} vs {engine_b_name}[/bold]\n")

        pbar = tqdm(total=cfg.sprt.max_games, desc="Games", unit="game")

        for round_num in range(max_rounds):
            # Get opening (cycle through if needed)
            opening_fen = openings[opening_idx % len(openings)]
            opening_idx += 1

            # Play game pair
            game1, game2 = runner.play_game_pair(engine_a, engine_b, opening_fen)
            all_games.extend([game1, game2])

            # Write games to PGN immediately (flushed to disk)
            if pgn_writer is not None:
                pgn_writer.write_games([game1, game2])

            # Update statistics
            for game in [game1, game2]:
                if game.result_value == 1.0:
                    wins += 1
                elif game.result_value == 0.0:
                    losses += 1
                else:
                    draws += 1

                # Track Syzygy adjudications
                if game.termination == GameTermination.SYZYGY_ADJUDICATED:
                    syzygy_adjudications += 1

            # Update SPRT
            sprt_result = sprt.update(wins, losses, draws)

            # Update progress bar
            pbar.update(2)
            pbar.set_postfix(
                W=wins,
                L=losses,
                D=draws,
                elo=f"{sprt_result.elo_estimate:+.1f}",
                llr=f"{sprt_result.llr:.2f}",
            )

            # Log to W&B
            if wandb_run:
                import wandb as wb

                wb.log(
                    {
                        "games": sprt_result.games,
                        "wins": wins,
                        "losses": losses,
                        "draws": draws,
                        "score": sprt_result.score,
                        "elo_estimate": sprt_result.elo_estimate,
                        "elo_error": sprt_result.elo_error,
                        "llr": sprt_result.llr,
                        "llr_lower": sprt_result.lower_bound,
                        "llr_upper": sprt_result.upper_bound,
                        "win_rate": sprt_result.win_rate,
                        "draw_rate": sprt_result.draw_rate,
                        "loss_rate": sprt_result.loss_rate,
                        "syzygy_adjudications": syzygy_adjudications,
                    }
                )

            # Check termination
            if sprt_result.status != SPRTStatus.CONTINUE:
                pbar.close()
                break

        else:
            pbar.close()

        # Final results
        final_result = sprt.update(wins, losses, draws)

        console.print("\n")
        console.print(Panel.fit(
            create_status_table(
                final_result,
                engine_a_name,
                engine_b_name,
                opening_idx,
                len(openings),
            ),
            title="[bold]Final Results[/bold]",
        ))

        # Print conclusion
        if final_result.status == SPRTStatus.H1_ACCEPTED:
            console.print(
                f"\n[bold green]✓ H1 Accepted:[/bold green] {engine_a_name} is "
                f"{cfg.sprt.elo1}+ Elo stronger than {engine_b_name}"
            )
        elif final_result.status == SPRTStatus.H0_ACCEPTED:
            console.print(
                f"\n[bold red]✗ H0 Accepted:[/bold red] {engine_a_name} is NOT "
                f"significantly stronger than {engine_b_name}"
            )
        else:
            console.print(
                f"\n[bold yellow]? Inconclusive:[/bold yellow] "
                f"Max games reached without conclusive result"
            )

        console.print(
            f"\nFinal Elo: [bold]{final_result.elo_estimate:+.1f} "
            f"± {final_result.elo_error:.1f}[/bold]"
        )

        # Log final result to W&B
        if wandb_run:
            import wandb as wb

            wb.summary["final_status"] = final_result.status.value
            wb.summary["final_elo"] = final_result.elo_estimate
            wb.summary["final_elo_error"] = final_result.elo_error
            wb.summary["final_games"] = final_result.games
            wb.summary["final_wins"] = wins
            wb.summary["final_losses"] = losses
            wb.summary["final_draws"] = draws
            wb.summary["final_syzygy_adjudications"] = syzygy_adjudications

        # Upload PGN to W&B if enabled
        if pgn_writer is not None and wandb_run:
            import wandb as wb

            wb.save(str(pgn_writer.path))

    finally:
        # Cleanup
        logger.info("Closing engines...")
        engine_a.close()
        engine_b.close()

        # Close game runner (releases tablebase file handles)
        runner.close()

        # Close PGN writer
        if pgn_writer is not None:
            pgn_writer.close()

        if wandb_run:
            import wandb as wb

            wb.finish()

    console.print("\n[bold green]Tournament complete![/bold green]")


if __name__ == "__main__":
    main()
