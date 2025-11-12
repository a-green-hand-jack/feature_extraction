# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a peptide stability feature extraction project for SIF (Simulated Intestinal Fluid) and SGF (Simulated Gastric Fluid) datasets. The project extracts molecular features from SMILES strings and generates exploratory data analysis (EDA) visualizations for peptide stability modeling.

**Key capabilities:**
- Extract molecular features using RDKit (Morgan/Avalon fingerprints, QED properties, physicochemical descriptors)
- Process CSV files with columns: `id`, `SMILES`, `SIF_class`, `SGF_class`
- Generate comprehensive EDA visualizations
- Handle missing labels (encoded as -1) throughout the pipeline

## Environment Management

**CRITICAL:** This project uses `uv` for Python environment management.

**Always use `uv run` to execute Python scripts:**
```bash
uv run python scripts/extract_sif_sgf_features.py [args]
uv run python scripts/visualize_data.py [args]
```

**Never use:**
- `python script.py` directly
- `pip install` for dependencies

**Setup commands:**
```bash
# Sync dependencies (after pulling changes)
uv sync

# Add a new dependency
uv add package-name
```

## Project Structure

```
feature_extraction/
├── data/
│   ├── raw/              # Original CSV files
│   └── cleaned/          # Cleaned CSV files (preprocessed)
├── outputs/
│   ├── features/         # NPZ files with extracted features
│   ├── figures/          # EDA visualization plots
│   └── class_distribution/  # Class distribution summaries
├── src/feature_extraction/
│   ├── featurizer.py     # PeptideFeaturizer class
│   ├── visualization.py  # DataVisualizer class
│   └── utils.py          # Helper functions
└── scripts/              # Executable scripts
    ├── clean_csv_data.py
    ├── extract_sif_sgf_features.py
    ├── visualize_data.py
    ├── visualize_class.py
    └── compare_feature_distributions.py
```

## Common Workflows

### 1. Feature Extraction Pipeline

**Extract features from cleaned CSV files:**
```bash
uv run python scripts/extract_sif_sgf_features.py \
    --input data/cleaned/sif_sgf_second_cleaned.csv \
    --output outputs/features/sif_sgf_second.npz
```

**Batch process all CSVs in a directory:**
```bash
uv run python scripts/extract_features.py \
    --input_dir data/cleaned/ \
    --output_dir outputs/features/
```

### 2. Data Visualization

**Generate EDA plots from NPZ files:**
```bash
uv run python scripts/visualize_data.py \
    --input_dir outputs/features/ \
    --output_dir outputs/figures/ \
    --dpi 300 \
    --format png
```

**Visualize class distribution:**
```bash
uv run python scripts/visualize_class.py \
    --input outputs/features/sif_sgf_second.npz \
    --output_dir outputs/class_distribution/
```

**Compare feature distributions:**
```bash
uv run python scripts/compare_feature_distributions.py \
    --input_dir outputs/features/ \
    --output_dir outputs/figures/
```

### 3. Data Cleaning

**Clean raw CSV files:**
```bash
uv run python scripts/clean_csv_data.py \
    --input data/raw/sif_sgf_second.csv \
    --output data/cleaned/sif_sgf_second_cleaned.csv
```

## Core Components

### PeptideFeaturizer (`src/feature_extraction/featurizer.py`)

**Purpose:** Extract molecular features from SMILES strings.

**Key features extracted:**
1. **QED Properties (8 features):** MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
2. **Physicochemical Descriptors (11 features):** Molecular weight, LogP, HBA, HBD, TPSA, rotatable bonds, ring count, Fsp3, heavy atom count, total atoms, rigidity proxy
3. **Gasteiger Charge Statistics (5 features):** Mean, max, min, std, sum
4. **Morgan Fingerprint (1024 bits by default)**
5. **Avalon Fingerprint (512 bits, optional):** Depends on RDKit build

**Usage:**
```python
from src.feature_extraction.featurizer import PeptideFeaturizer

featurizer = PeptideFeaturizer(
    morgan_bits=1024,
    avalon_bits=512,
    use_avalon=True
)

features, success = featurizer.featurize("CC(C)C[C@H](NC(=O)...")
feature_names = featurizer.get_feature_names()
```

**Important notes:**
- Returns `(None, False)` for invalid SMILES
- All NaN/Inf values are safely converted to 0.0
- Feature order: QED → Physchem → Gasteiger → Morgan → Avalon

### DataVisualizer (`src/feature_extraction/visualization.py`)

**Purpose:** Generate EDA visualizations from NPZ feature files.

**Key visualizations:**
1. **Feature distributions:** Histograms with KDE for MW, LogP, HBA, HBD
2. **Label distribution:** Bar charts for SIF/SGF class counts (excluding -1)
3. **Joint distribution:** Heatmap showing SIF vs SGF class co-occurrence (including -1 as "Missing")
4. **Grouped boxplots:** MW and LogP vs stability classes
5. **Correlation heatmap:** Pearson correlation for features
6. **Scatter matrix:** Pairwise feature relationships

**Missing label handling:**
- Labels with value `-1` indicate missing/unavailable data
- Bar charts filter out -1 and show missing count in title
- Heatmaps include -1 as "Missing" category

**Usage:**
```python
from src.feature_extraction.visualization import DataVisualizer

visualizer = DataVisualizer(
    X=features,
    y_sif=sif_labels,
    y_sgf=sgf_labels,
    feature_names=feature_names,
    dataset_name="SIF_SGF_Dataset",
    dpi=300
)

visualizer.plot_feature_distributions(output_dir, format="png")
visualizer.plot_label_distribution(output_dir, format="png")
summary = visualizer.generate_summary_statistics()
```

## Data Format

### Input CSV Format
```csv
id,SMILES,SIF_class,SGF_class
peptide_001,CC(C)C[C@H](NC...)...,0,1
peptide_002,C[C@H](NC(=O)...)...,-1,2
```

**Column requirements:**
- `id`: Unique identifier (string/int)
- `SMILES`: Valid SMILES string
- `SIF_class`: Integer class label or -1 (missing)
- `SGF_class`: Integer class label or -1 (missing)

### Output NPZ Format

**Saved arrays:**
- `X`: Feature matrix, shape (n_samples, n_features), dtype float32
- `y_sif`: SIF class labels, shape (n_samples,), dtype int
- `y_sgf`: SGF class labels, shape (n_samples,), dtype int
- `ids`: Sample IDs, shape (n_samples,), dtype object
- `feature_names`: Feature names, shape (n_features,), dtype object
- `mask_valid`: Boolean mask of valid samples, shape (n_rows,), dtype bool
- `metadata`: Dictionary with extraction metadata, dtype object

**Loading NPZ files:**
```python
import numpy as np

npz = np.load("outputs/features/dataset.npz", allow_pickle=True)
X = npz['X']
y_sif = npz['y_sif']
y_sgf = npz['y_sgf']
feature_names = npz['feature_names']
ids = npz['ids']
```

## Key Implementation Details

### Missing Label Handling

**Throughout the codebase:**
- Missing labels are encoded as `-1`
- `-1` is a valid value preserved during feature extraction
- Visualization scripts handle `-1` consistently:
  - Bar charts: Filter out -1, show missing count
  - Heatmaps: Include -1 as "Missing" category
  - Statistics: Report missing counts separately

### Feature Naming Convention

**Prefixes indicate feature type:**
- `QED_*`: QED properties (e.g., `QED_MW`, `QED_ALOGP`)
- `PC_*`: Physicochemical descriptors (e.g., `PC_MolWt`, `PC_LogP`)
- `GC_*`: Gasteiger charge statistics (e.g., `GC_Mean`, `GC_Max`)
- `Morgan_*`: Morgan fingerprint bits (e.g., `Morgan_0`, `Morgan_1`)
- `Avalon_*`: Avalon fingerprint bits (e.g., `Avalon_0`, `Avalon_1`)

### Error Handling

**All feature extraction functions:**
- Use try-except blocks with fallback to zero vectors
- Log warnings for individual failures (debug level)
- Never raise exceptions that stop batch processing
- Return success flags: `(features, True)` or `(None, False)`

## Research Context

This project supports peptide stability modeling in simulated gastrointestinal fluids. Key research focuses from `docs/Visualization/README.md`:

**Important physicochemical properties:**
- **Molecular Weight (MW):** Size/permeability indicator
- **LogP:** Lipophilicity/hydrophobicity
- **HBA/HBD:** Hydrogen bonding capacity
- **Rigidity proxy:** Rings / (1 + rotatable bonds)

**Performance metrics for classification:**
- Accuracy, Precision, Recall, F1-score, AUC
- Threshold variants: SIF1/SGF1 (6h), SIF2/SGF2 (2h)

**Feature importance insights:**
- Lipophilicity (LogP-related features) are highly predictive
- Molecular size and structural rigidity are key determinants
- Morgan/Avalon fingerprints capture structural patterns

## Development Notes

### When adding new features:

1. Add extraction logic to `PeptideFeaturizer.featurize()`
2. Update `PeptideFeaturizer.get_feature_names()` to include new names
3. Update `PeptideFeaturizer.n_features` property
4. Use consistent naming convention with prefixes
5. Ensure error handling returns default values (not None)

### When modifying visualizations:

1. Check if changes affect missing label handling (-1 values)
2. Update both bar charts (filter -1) and heatmaps (include -1)
3. Maintain consistent labeling: "Missing" for -1 in visualizations
4. Test with datasets containing various missing patterns

### When debugging:

**Check log files in project root:**
- `feature_extraction.log`: Feature extraction details
- `visualization.log`: Visualization generation details
- `visualize_class.log`: Class distribution analysis

**Common issues:**
- Avalon not available: Check RDKit build, set `use_avalon=False`
- Invalid SMILES: Check `mask_valid` array in NPZ output
- Missing features: Verify feature names match between extraction and visualization
