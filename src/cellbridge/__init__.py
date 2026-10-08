"""Public CellBridge interface.

The solver is the frozen v1.0 implementation in ``responsebridge.cellbridge``.
This package adds the fit, predict, contribution and interval calls used by
the tutorial.
"""
from .api import FittedCellBridge, fit, fit_anndata, panel_curve

__all__ = ["FittedCellBridge", "fit", "fit_anndata", "panel_curve"]
__version__ = "1.1.0"
