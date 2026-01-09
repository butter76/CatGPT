# Bugbot Review Guidelines for CatGPT

## Project Context

CatGPT is an ML research project focused on chess and deep learning. The codebase uses:
- Python 3.12 with modern type hints
- PyTorch for deep learning
- Hydra/OmegaConf for configuration
- python-chess for chess logic
- ruff for linting and formatting
- pyright for type checking

## Code Quality Standards

### Type Hints
- All functions MUST have complete type annotations
- Use `| None` syntax instead of `Optional[]`
- Use modern generic syntax (e.g., `list[str]` not `List[str]`)

### Error Handling
- Use custom exception messages with context
- Prefer early returns for validation
- Log errors with loguru before raising

### Performance
- Flag any code that loads large data in __init__ without lazy loading
- Watch for inefficient tensor operations (e.g., repeated .to(device) calls)
- Check for missing torch.no_grad() in evaluation code

### Testing
- Tests should be fast by default
- GPU-requiring tests must be marked with @pytest.mark.gpu
- Slow tests must be marked with @pytest.mark.slow

## Things to Ignore

- Don't flag TODOs - they're intentional placeholders
- Don't suggest adding docstrings to test methods
- Don't suggest changes to configuration files unless there's a bug
- Backwards compatibility is NOT valued - prefer clean rewrites over fallbacks

## Priority Issues

Flag these as HIGH priority:
- Missing type annotations on public functions
- Potential memory leaks (especially in training loops)
- Race conditions in data loading
- Hardcoded paths or credentials
- Missing .detach() on tensors used for logging
