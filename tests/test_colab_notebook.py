"""
Tests for Colab notebook compatibility and data requirements.

Run before pushing notebooks to Colab to catch common issues.
"""

import pytest
import json
from pathlib import Path
import pandas as pd
import nbformat
from nbformat.v4 import new_notebook, new_code_cell


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
COLAB_DIR = PROJECT_ROOT / 'colab'
COLAB_SPLITS_DIR = COLAB_DIR / 'data_splits'
COLAB_NOTEBOOK = COLAB_DIR / '07_transfer_learning_colab.ipynb'


class TestColabDataSplits:
    """Test Colab-specific data split files."""

    def test_splits_directory_exists(self):
        """Colab data_splits directory must exist."""
        assert COLAB_SPLITS_DIR.exists(), f"Missing: {COLAB_SPLITS_DIR}"

    def test_required_files_exist(self):
        """All required CSV and config files must exist."""
        required_files = [
            'train_split.csv',
            'val_split.csv',
            'test_split.csv',
            'preprocessing_config.json'
        ]

        for filename in required_files:
            file_path = COLAB_SPLITS_DIR / filename
            assert file_path.exists(), f"Missing required file: {filename}"

    def test_csv_has_image_index_column(self):
        """CSV files must have 'Image Index' column (not 'full_path')."""
        for split_name in ['train', 'val', 'test']:
            csv_path = COLAB_SPLITS_DIR / f'{split_name}_split.csv'
            df = pd.read_csv(csv_path)

            assert 'Image Index' in df.columns, \
                f"{split_name}_split.csv missing 'Image Index' column"

            # Should NOT have full_path (Colab builds it dynamically)
            assert 'full_path' not in df.columns, \
                f"{split_name}_split.csv should not have 'full_path' (built at runtime)"

    def test_csv_has_disease_columns(self):
        """CSV files must have disease label columns."""
        expected_diseases = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
            'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
            'Pneumonia', 'Pneumothorax'
        ]

        train_df = pd.read_csv(COLAB_SPLITS_DIR / 'train_split.csv')

        for disease in expected_diseases:
            assert disease in train_df.columns, \
                f"Missing disease column: {disease}"

    def test_csv_disease_values_are_binary(self):
        """Disease columns should be 0 or 1."""
        train_df = pd.read_csv(COLAB_SPLITS_DIR / 'train_split.csv')

        disease_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
            'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
            'Pneumonia', 'Pneumothorax'
        ]

        for col in disease_columns:
            unique_vals = set(train_df[col].unique())
            assert unique_vals.issubset({0, 1}), \
                f"{col} has non-binary values: {unique_vals}"

    def test_preprocessing_config_valid(self):
        """preprocessing_config.json must be valid JSON with required keys."""
        config_path = COLAB_SPLITS_DIR / 'preprocessing_config.json'

        with open(config_path) as f:
            config = json.load(f)

        assert 'disease_classes' in config, "Missing 'disease_classes' key"
        assert 'num_classes' in config, "Missing 'num_classes' key"
        assert len(config['disease_classes']) == config['num_classes'], \
            "disease_classes length != num_classes"
        assert config['num_classes'] == 14, "Should have 14 disease classes"

    def test_csv_files_are_smaller_than_kaggle(self):
        """Colab CSVs should be significantly smaller (no full_path column)."""
        # Colab train should be < 10 MB (actual ~4.7 MB)
        train_size = (COLAB_SPLITS_DIR / 'train_split.csv').stat().st_size / 1024 / 1024
        assert train_size < 10, f"train_split.csv too large: {train_size:.1f} MB (should be ~4-5 MB)"

        # Val and test should be < 2 MB each
        val_size = (COLAB_SPLITS_DIR / 'val_split.csv').stat().st_size / 1024 / 1024
        test_size = (COLAB_SPLITS_DIR / 'test_split.csv').stat().st_size / 1024 / 1024

        assert val_size < 2, f"val_split.csv too large: {val_size:.1f} MB"
        assert test_size < 2, f"test_split.csv too large: {test_size:.1f} MB"


class TestColabNotebook:
    """Test Colab notebook structure and cell dependencies."""

    @pytest.fixture
    def notebook(self):
        """Load Colab notebook."""
        with open(COLAB_NOTEBOOK) as f:
            return nbformat.read(f, as_version=4)

    def test_notebook_exists(self):
        """Colab notebook file must exist."""
        assert COLAB_NOTEBOOK.exists(), f"Missing: {COLAB_NOTEBOOK}"

    def test_notebook_has_cells(self, notebook):
        """Notebook must have cells."""
        assert len(notebook.cells) > 0, "Notebook has no cells"

    def test_path_building_before_generators(self, notebook):
        """Cell that builds 'full_path' must come before data generators."""
        path_building_idx = None
        generator_idx = None

        for idx, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                source = ''.join(cell.source)

                # Find cell that builds full_path
                if "build_image_path" in source and "train_df['full_path']" in source:
                    path_building_idx = idx

                # Find cell that creates ImageDataGenerator
                if "ImageDataGenerator" in source and "x_col='full_path'" in source:
                    generator_idx = idx

        assert path_building_idx is not None, \
            "Notebook missing cell that builds 'full_path' column"
        assert generator_idx is not None, \
            "Notebook missing cell that creates data generators"
        assert path_building_idx < generator_idx, \
            f"Path building (cell {path_building_idx}) must come before generators (cell {generator_idx})"

    def test_no_hardcoded_paths(self, notebook):
        """Notebook should not contain hardcoded local paths."""
        forbidden_patterns = [
            '/Volumes/SSD/',
            '/Users/',
            'C:\\',
            'D:\\',
        ]

        for idx, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                source = ''.join(cell.source)
                for pattern in forbidden_patterns:
                    assert pattern not in source, \
                        f"Cell {idx} contains hardcoded path: {pattern}"

    def test_uses_image_index_not_full_path_for_loading(self, notebook):
        """Notebook should load CSVs and build full_path from Image Index."""
        found_image_index_usage = False

        for cell in notebook.cells:
            if cell.cell_type == 'code':
                source = ''.join(cell.source)

                # Should use 'Image Index' to build paths
                if "'Image Index'" in source and "build_image_path" in source:
                    found_image_index_usage = True

        assert found_image_index_usage, \
            "Notebook should build paths from 'Image Index' column"

    def test_has_cache_check_for_persistent_disk(self, notebook):
        """Notebook should check for cached dataset (Colab Pro)."""
        has_cache_check = False

        for cell in notebook.cells:
            if cell.cell_type == 'code':
                source = ''.join(cell.source)

                # Look for cache check logic
                if "CACHE_DIR" in source and "exists()" in source:
                    has_cache_check = True

        assert has_cache_check, \
            "Notebook should check for cached data (Colab Pro persistent disk)"

    def test_no_colab_builtin_auth(self, notebook):
        """Should use custom OAuth, not Colab's auth.authenticate_user()."""
        for idx, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                source = ''.join(cell.source)

                # Should NOT use Colab's built-in auth (shows "third party" warning)
                assert "auth.authenticate_user()" not in source, \
                    f"Cell {idx} uses Colab built-in auth (shows 'third party' warning). Use client_secrets.json instead."


class TestPathBuildingLogic:
    """Test the path building function that maps filenames to full paths."""

    def test_build_image_path_function(self):
        """Test path building logic in isolation."""
        from pathlib import Path

        # Simulate the function from the notebook
        def build_image_path(filename, base_dir):
            """Find image file in the downloaded Kaggle dataset."""
            for subdir in sorted(Path(base_dir).glob('images_*')):
                if subdir.is_dir():
                    img_path = subdir / 'images' / filename
                    if img_path.exists():
                        return str(img_path)
            return None

        # Test with mock directory structure (if available)
        # This is a smoke test - actual test needs real data
        test_dir = PROJECT_ROOT / 'data' / 'raw'
        if test_dir.exists():
            test_filename = '00000001_000.png'
            result = build_image_path(test_filename, test_dir)

            # Should either find it or return None (not crash)
            assert result is None or isinstance(result, str)


class TestDataIntegrity:
    """Test data integrity across splits."""

    def test_no_patient_overlap_between_splits(self):
        """Train, val, test should have no overlapping patients."""
        train_df = pd.read_csv(COLAB_SPLITS_DIR / 'train_split.csv')
        val_df = pd.read_csv(COLAB_SPLITS_DIR / 'val_split.csv')
        test_df = pd.read_csv(COLAB_SPLITS_DIR / 'test_split.csv')

        train_patients = set(train_df['Patient ID'].unique())
        val_patients = set(val_df['Patient ID'].unique())
        test_patients = set(test_df['Patient ID'].unique())

        assert len(train_patients & val_patients) == 0, \
            "Train and val splits have overlapping patients"
        assert len(train_patients & test_patients) == 0, \
            "Train and test splits have overlapping patients"
        assert len(val_patients & test_patients) == 0, \
            "Val and test splits have overlapping patients"

    def test_all_splits_have_data(self):
        """All splits should have reasonable number of samples."""
        train_df = pd.read_csv(COLAB_SPLITS_DIR / 'train_split.csv')
        val_df = pd.read_csv(COLAB_SPLITS_DIR / 'val_split.csv')
        test_df = pd.read_csv(COLAB_SPLITS_DIR / 'test_split.csv')

        # NIH dataset has 112,120 images total
        # Train should be ~70%, val ~15%, test ~15%
        assert len(train_df) > 70000, f"Train split too small: {len(train_df)}"
        assert len(val_df) > 10000, f"Val split too small: {len(val_df)}"
        assert len(test_df) > 10000, f"Test split too small: {len(test_df)}"

    def test_image_filenames_are_valid(self):
        """Image Index should contain valid filenames."""
        train_df = pd.read_csv(COLAB_SPLITS_DIR / 'train_split.csv')

        # Sample 100 filenames
        sample_filenames = train_df['Image Index'].sample(min(100, len(train_df)))

        for filename in sample_filenames:
            # Should be PNG files
            assert filename.endswith('.png'), f"Invalid filename: {filename}"
            # Should match NIH naming pattern (8 digits_NNN.png, 16-17 chars)
            assert 16 <= len(filename) <= 18, f"Unexpected filename format: {filename}"


class TestOAuthSetup:
    """Test OAuth credential setup files."""

    def test_client_secrets_example_exists(self):
        """Example OAuth credentials file should exist."""
        example_file = COLAB_DIR / 'client_secrets.json.example'
        assert example_file.exists(), \
            "Missing client_secrets.json.example for documentation"

    def test_client_secrets_example_is_valid_json(self):
        """Example file should be valid JSON."""
        example_file = COLAB_DIR / 'client_secrets.json.example'

        with open(example_file) as f:
            config = json.load(f)

        assert 'installed' in config, "Missing 'installed' key"
        assert 'client_id' in config['installed'], "Missing 'client_id'"
        assert 'client_secret' in config['installed'], "Missing 'client_secret'"

    def test_actual_client_secrets_not_committed(self):
        """Actual client_secrets.json should NOT be in git."""
        actual_file = COLAB_DIR / 'client_secrets.json'

        # File might exist locally, but should be gitignored
        # Check .gitignore includes it
        gitignore = PROJECT_ROOT / '.gitignore'
        with open(gitignore) as f:
            gitignore_content = f.read()

        assert 'colab/client_secrets.json' in gitignore_content, \
            "client_secrets.json should be in .gitignore"


# Pytest markers for organization
pytest.mark.colab = pytest.mark.mark('colab')
pytest.mark.data = pytest.mark.mark('data')


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
