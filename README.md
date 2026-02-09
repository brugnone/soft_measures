# FCM Scoring

A Python utility for comparing and scoring two Fuzzy Cognitive Maps (FCMs) using semantic similarity embeddings.

## Features

- **Simple API**: Clean function-based interface for comparing FCMs
- **Flexible Input/Output**: Support CSV and JSON formats for both input and output
- **Semantic Matching**: Uses transformer embeddings to match nodes between FCMs
- **Comprehensive Metrics**: F1, Jaccard, and detailed edge matching statistics
- **Memory Efficient**: Configurable batch processing
- **Reproducible**: Fixed random seeds for consistency

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

Clone and install dependencies:

```bash
git clone <repo-url>
cd soft_measures
pip install -r requirements.txt
```

Dependencies include:
- `pandas` - Data handling
- `torch` - Deep learning
- `transformers` - Pre-trained language models for embeddings
- `scipy` - Scientific computing (Hungarian algorithm for matching)
- `numpy` - Numerical computing

## Quick Start

### As a Python Function

```python
from score_fcms import score_fcm

results = score_fcm(
    fcm1_path="path/to/fcm1.csv",
    fcm2_path="path/to/fcm2.json",
    output_format="csv"
)

print(f"F1 Score: {results['F1'].iloc[0]:.4f}")
print(f"Jaccard Score: {results['Jaccard'].iloc[0]:.4f}")
```

### From Command Line

```bash
python score_fcms.py path/to/fcm1.csv path/to/fcm2.json
```

Optional arguments:
```bash
python score_fcms.py fcm1.csv fcm2.json \
    --threshold 0.6 \
    --model-name "Qwen/Qwen3-Embedding-0.6B" \
    --output-format both \
    --batch-size 2 \
    --output-dir ./results
```

### Batch Comparison of Directories

Compare all matching FCM files from two directories:

```python
from compare_fcm_directories import compare_directories

results = compare_directories(
    dir1="path/to/first_set",
    dir2="path/to/second_set",
    output_dir="comparison_results"
)

print(f"Average F1: {results['F1'].mean():.4f}")
```

Or from command line:
```bash
python compare_fcm_directories.py dir1/ dir2/ --output-dir results/
```

This will:
- Find all CSV/JSON files with matching names in both directories
- **Load model weights once** and reuse for all pairs (much faster than individual scoring)
- Score each pair
- Save individual results in subdirectories
- Generate a combined summary CSV/JSON

## Input Formats

### CSV Format (Adjacency Matrix)
A square matrix with node names as both index and columns.

- **Diagonal** should be 0 (no self-loops)
- **Values**: positive (reinforcing) or negative (inhibiting)
- **Empty cells** are treated as 0 (no relationship)

### JSON Format (Edge List)
Edge-based representation with source, target, and weight:

```json
{
  "edges": [
    {
      "source": "node_0",
      "target": "node_1",
      "weight": 0.8,
    },
    {
      "source": "node_1",
      "target": "node_2",
      "weight": -0.5,
    }
  ]
}
```

## Output

Results are saved in the specified format(s):

### CSV Output
Saved as `<filename>_scoring_results.csv` with columns:

| Column | Description |
|--------|-------------|
| `F1` | F1 score with scaled true/partial positives |
| `Jaccard` | Jaccard similarity index |
| `TP` | True positives |
| `PP` | Partial positives (sign mismatch) |
| `FP` | False positives |
| `FN` | False negatives |
| `threshold` | Matching threshold used |
| `tp_scale` | True positive scaling factor |
| `pp_scale` | Partial positive scaling factor |
| `fcm1_nodes` / `fcm2_nodes` | Node counts |
| `fcm1_edges` / `fcm2_edges` | Edge counts |

### JSON Output
Same data structure in JSON format: `<filename>_scoring_results.json`

## Configuration

### Threshold
- Controls matching confidence (0.0 to 1.0)
- Default: 0.6
- Higher = stricter matching, fewer matches

### Scaling Factors
- `tp_scale`: Weight for true positive matches (default: 1.0)
- `pp_scale`: Weight for partial positive matches (default: 1.1)
- Affects F1 calculation, not Jaccard

### Batch Size
- Reduces memory usage on low-VRAM systems
- Default: 2
- Lower values = less memory, slower processing

### Models
Supported embedding models (any HuggingFace model):
- `Qwen/Qwen3-Embedding-0.6B` (default, fast & efficient)
- `intfloat/e5-base`
- `intfloat/multilingual-e5-large`

## Examples

See [examples/](examples/) for complete working examples with sample data.

Run the basic example:
```bash
cd examples
python example.py
```

## Jupyter Notebook

A detailed walkthrough is available in `examples/fcm_scoring_walkthrough.ipynb` that demonstrates:
- Loading FCM data (CSV and JSON)
- Basic scoring with default parameters
- Parameter tuning and threshold analysis
- Interpreting results

## Performance Tips

1. **GPU Usage**: Automatically uses CUDA if available (falls back to CPU)
2. **Memory**: Reduce `--batch-size` if you hit OOM errors
3. **Speed**: Increase `--batch-size` if you have spare VRAM
4. **Models**: Smaller models are faster (Qwen3-0.6B is recommended)

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```bash
python score_fcms.py fcm1.csv fcm2.json --batch-size 1
```

### Poor Matching Results
- Try adjusting threshold (lower = more matches, higher = stricter)
- Experiment with different embedding models
- Ensure node names are consistent between FCMs

### File Format Issues
- CSV must have node names as index (first column)
- JSON must have an "edges" array with "source", "target", and "weight" fields
- Use `.csv` or `.json` file extensions for automatic format detection

## API Reference

### `score_fcm(fcm1_path, fcm2_path, ...)`

```python
score_fcm(
    fcm1_path: str,                              # Path to first FCM
    fcm2_path: str,                              # Path to second FCM
    output_dir: Optional[str] = None,            # Output directory
    output_format: str = "csv",                  # "csv", "json", or "both"
    threshold: float = 0.6,                      # Similarity threshold
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",  # Embedding model
    tp_scale: float = 1.0,                       # TP scale factor
    pp_scale: float = 1.1,                       # PP scale factor
    batch_size: int = 2,                         # Processing batch size
    seed: int = 42,                              # Random seed
    verbose: bool = True                         # Print progress
) -> pd.DataFrame
```

Returns a DataFrame with scoring results.

### `score_fcm_with_scorer(fcm1_path, fcm2_path, scorer, ...)`

```python
score_fcm_with_scorer(
    fcm1_path: str,                              # Path to first FCM
    fcm2_path: str,                              # Path to second FCM
    scorer: ScoreCalculator,                     # Pre-initialized scorer instance
    output_dir: Optional[str] = None,            # Output directory
    output_format: str = "csv",                  # "csv", "json", or "both"
    verbose: bool = True                         # Print progress
) -> pd.DataFrame
```

Scores two FCMs using a pre-initialized ScoreCalculator. **Much faster for multiple pairs** as it reuses loaded model weights.

Example for batch processing:
```python
from score_fcms import ScoreCalculator, score_fcm_with_scorer

# Initialize scorer once
scorer = ScoreCalculator(threshold=0.6, model_name="Qwen/Qwen3-Embedding-0.6B", 
                         data="batch", tp_scale=1.0, pp_scale=1.1, seed=42)
scorer.batch_size = 2

# Reuse for multiple pairs
for fcm1, fcm2 in fcm_pairs:
    result = score_fcm_with_scorer(fcm1, fcm2, scorer)
```

### `compare_directories(dir1, dir2, ...)`

```python
compare_directories(
    dir1: str,                                   # First directory with FCM files
    dir2: str,                                   # Second directory with FCM files
    output_dir: Optional[str] = None,            # Output directory (default: comparison_results)
    output_format: str = "both",                 # "csv", "json", or "both"
    threshold: float = 0.5,                      # Similarity threshold
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",  # Embedding model
    tp_scale: float = 1.0,                       # TP scale factor
    pp_scale: float = 0.6,                       # PP scale factor
    batch_size: int = 2,                         # Processing batch size
    seed: int = 42,                              # Random seed
    verbose: bool = True                         # Print progress
) -> pd.DataFrame
```

Compares all FCM files with matching names from two directories. Returns a combined DataFrame with results for all pairs.

<!-- ## License

[Specify your license here]

## Contact

For questions or issues, contact [your contact information] -->
