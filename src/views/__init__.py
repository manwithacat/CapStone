"""
Views module for standalone page views in the Streamlit app.

Views are full-page displays that are separate from the main dashboard tabs.
They have their own routing and navigation logic.
"""

from src.views.radiology_guide import render_radiology_guide_tab

__all__ = [
    'render_radiology_guide_tab',
]
