"""
Filtering modules for pattern-aware text preprocessing.

Includes:
- heuristic: Basic heuristic filters (e.g., nopunc filtering)
- refinedweb: RefinedWeb-style heuristic filters from DCLM
- quality_classifier: FastText-based quality classification (DCLM)
"""

from .heuristic import nopunc_filtering
from .refinedweb import apply_refinedweb_heuristics
from .quality_classifier import predict_quality, process_file as filter_file_quality

__all__ = [
    'nopunc_filtering',
    'apply_refinedweb_heuristics',
    'predict_quality',
    'filter_file_quality',
]
