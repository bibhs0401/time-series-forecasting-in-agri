'''Model package: statistical baselines (always available) and PyTorch ST-GNNs.

Baselines are pure NumPy/statsmodels and always importable. The GNN models
need PyTorch; that import is guarded so 'import src.models' never fails on
a machine without torch. TORCH_AVAILABLE reports whether the ST-GNNs and
build_model are usable.
'''

from src.models.baselines import (
    BASELINES, make_baseline,
    SeasonalNaive, Naive, ETS, ARIMA,
)

try:
    import torch  # noqa: F401
    from src.models.gnn import (
        build_model, TORCH_MODELS,
        NoGraphBackbone, StaticGraphGNN, StaticGraphGATGNN, IdentityGraphGNN,
        RelationalMultiplexGNN, RelationalMultiplexGATGNN, AdaptiveGraphGNN,
    )
    TORCH_AVAILABLE = True
except Exception as _e:   # torch missing or failed to load
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _e

    def build_model(*a, **k):   # type: ignore
        raise ImportError(
            'PyTorch is required for the GNN models but is not available: '
            f'{_TORCH_IMPORT_ERROR}'
        )

    TORCH_MODELS = ()

__all__ = [
    'BASELINES', 'make_baseline',
    'SeasonalNaive', 'Naive', 'ETS', 'ARIMA',
    'build_model', 'TORCH_MODELS', 'TORCH_AVAILABLE',
    'NoGraphBackbone', 'StaticGraphGNN', 'StaticGraphGATGNN', 'IdentityGraphGNN',
    'RelationalMultiplexGNN', 'RelationalMultiplexGATGNN', 'AdaptiveGraphGNN',
]
