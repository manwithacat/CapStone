"""
Tab Modules - Modular Streamlit Tab Components

This package contains the tab rendering functions for the NIH Chest X-Ray
Disease Detection dashboard. Each tab is a separate module for better
code organization and maintainability.

============================================================================
WHAT IS THIS FILE? (__init__.py)
============================================================================

This special file makes the 'tabs' folder into a Python PACKAGE (not just a folder).

Key concepts:
1. Any folder with an __init__.py file becomes a Python package
2. This allows us to import from the folder: from src.tabs import ...
3. Without __init__.py, Python would treat 'tabs' as just a regular folder
4. The double underscores (dunder) mean this is a "magic" Python file with special meaning

WHY DO WE NEED THIS?
- It allows clean imports in app.py: from src.tabs import render_data_exploration_tab
- Instead of: from src.tabs.data_exploration import render_data_exploration_tab (more verbose)
- It centralizes all our tab imports in one place
- It makes the code more maintainable and easier to refactor

============================================================================
"""

# ============================================================================
# IMPORTS - Bring in functions from our tab modules
# ============================================================================
# The dot (.) before each module name means "from THIS package" (relative import)
# This is called a RELATIVE IMPORT and it's the recommended way to import within a package

# Example: "from .data_exploration" means "from the data_exploration.py file in THIS folder"
# We then specify which function to import: "import render_data_exploration_tab"

# Why relative imports (.data_exploration) instead of absolute (src.tabs.data_exploration)?
# - They work regardless of where the package is installed
# - They're more concise and easier to read
# - They make the package more portable and reusable

from .data_exploration import render_data_exploration_tab  # Tab 1: Dataset overview and demographics
from .hypothesis_validation import render_hypothesis_validation_tab  # Tab 2: Statistical hypothesis tests
from .model_performance import render_model_performance_tab  # Tab 3: Model evaluation metrics
from .disease_detector import render_disease_detector_tab  # Tab 4: Interactive disease prediction
from .clinical_insights import render_clinical_insights_tab  # Tab 5: Key findings and recommendations

# ============================================================================
# __all__ - Define what gets exported from this package
# ============================================================================
# __all__ is a list of strings defining the "public API" of this package
# When someone writes: from src.tabs import *
# Python will ONLY import the names listed in __all__

# Benefits of defining __all__:
# 1. Documentation: Shows users what functions they should use
# 2. Clean namespace: Prevents internal helper functions from being imported
# 3. IDE support: Helps code editors provide better autocomplete
# 4. Explicit is better than implicit: Clear about what's meant to be used externally

# In our case, we export all 5 render functions since they're all meant to be used by app.py
__all__ = [
    "render_data_exploration_tab",
    "render_hypothesis_validation_tab",
    "render_model_performance_tab",
    "render_disease_detector_tab",
    "render_clinical_insights_tab",
]

# ============================================================================
# HOW THIS WORKS IN PRACTICE
# ============================================================================
# In app.py, we can now write:
#   from src.tabs import render_data_exploration_tab, render_hypothesis_validation_tab, ...
#
# Instead of having to write:
#   from src.tabs.data_exploration import render_data_exploration_tab
#   from src.tabs.hypothesis_validation import render_hypothesis_validation_tab
#   from src.tabs.model_performance import render_model_performance_tab
#   from src.tabs.disease_detector import render_disease_detector_tab
#   from src.tabs.clinical_insights import render_clinical_insights_tab
#
# This __init__.py file acts as a "hub" that collects all the functions
# and makes them available in one convenient import statement!
