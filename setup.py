"""
Setup configuration for NIH Chest X-Ray Disease Detection project.

This makes the src/ directory a proper Python package that can be imported
without sys.path modifications.

Installation:
    pip install -e .  # Editable/development mode (recommended)
    pip install .     # Standard installation
"""

from setuptools import setup, find_packages

setup(
    name="chest-xray-detection",
    version="0.1.0",
    description="NIH Chest X-Ray Disease Detection - Multi-label Classification",
    author="Code Institute Capstone Project",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "kaleido>=0.2.1",
        "streamlit>=1.28.0",
        "altair>=5.0.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "scipy>=1.11.0",
        "tensorflow>=2.15.0,<2.19.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.22.0",
        "albumentations>=1.3.0",
        "tf-keras-vis==0.8.7",
        "kagglehub>=0.2.0",
        "kaggle>=1.6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipykernel>=6.25.0",
            "nbstripout>=0.6.0",
            "nbdime>=4.0.0",
            "tqdm>=4.65.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
            "pyright>=1.1.350",  # For type checking (Pylance backend)
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
