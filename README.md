# FCM Scoring

A Python utility for scoring generated Fuzzy Cognitive Maps (FCMs) against ground truth adjacency matrices using semantic similarity.

## Features

- **Semantic Similarity Matching**: Uses transformer embeddings to match nodes between generated and ground truth FCMs
- **Flexible Scoring**: Computes F1, Jaccard, and other metrics
- **Handle Partial Matches**: Distinguishes between true positives, partial positives, false positives, and false negatives
- **Configurable**: Adjustable thresholds, scaling factors, and embedding models
- **Memory Efficient**: Batch processing with configurable batch sizes
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
    gt_csv_path="path/to/ground_truth.csv",
    gen_json_path="path/to/generated_fcm.json"
)

print(f"F1 Score: {results['F1'].iloc[0]:.4f}")
print(f"Jaccard Score: {results['Jaccard'].iloc[0]:.4f}")
```

### From Command Line

```bash
python score_fcms.py path/to/ground_truth.csv path/to/generated_fcm.json
```

Optional arguments:
```bash
python score_fcms.py ground_truth.csv generated_fcm.json \
    --threshold 0.6 \
    --model-name "Qwen/Qwen3-Embedding-0.6B" \
    --tp-scale 1.0 \
    --pp-scale 1.1 \
    --batch-size 2 \
    --output-dir ./results
```

## Input Formats

### Ground Truth CSV
A square adjacency matrix with node names as both index and columns:

```csv
,cluster_0,cluster_1,cluster_2
cluster_0,0,1,-1
cluster_1,-1,0,0.5
cluster_2,0,-1,0
```

- **Diagonal** should be 0 (no self-loops)
- **Values**: positive (reinforcing) or negative (inhibiting)
- **Empty cells** are treated as 0 (no relationship)

### Generated FCM JSON
Edge-based representation with source, target, and weight:

```json
{
  "edges": [
    {
      "source": "cluster_0",
      "target": "cluster_1",
      "weight": 0.8,
      "confidence": 0.95,
      "type": "inter_cluster"
    },
    {
      "source": "cluster_1",
      "target": "cluster_2",
      "weight": -0.5,
      "confidence": 0.87,
      "type": "inter_cluster"
    }
  ]
}
```

### Metadata JSON (Optional)
Enriches cluster information with semantic context:

```json
{
  "clusters": [
    {
      "id": "cluster_0",
      "name": "stakeholder_engagement",
      "summary": "Level of engagement by stakeholders",
      "concepts": ["involvement", "participation", "feedback"]
    }
  ]
}
```

## Output

Results are saved to `<dataset>_scoring_results.csv` with columns:

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
| `gt_nodes` / `gen_nodes` | Node counts |
| `gt_edges` / `gen_edges` | Edge counts |

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

## Example

See [examples/](examples/) for complete working examples with sample data.

Run the example:
```bash
cd examples
python example.py
```

## Performance Tips

1. **GPU Usage**: Automatically uses CUDA if available (falls back to CPU)
2. **Memory**: Reduce `--batch-size` if you hit OOM errors
3. **Speed**: Increase `--batch-size` if you have spare VRAM
4. **Metadata**: Include metadata JSON to improve semantic matching

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```bash
python score_fcms.py ... --batch-size 1
```

### Poor Matching Results
- Increase threshold (too many false positives)
- Decrease threshold (too many false negatives)
- Try a different model: `--model-name "intfloat/e5-base"`
- Add metadata for better semantic context

### Missing Metadata
If metadata file isn't found, the scorer will look for `<dataset>_cluster_metadata.json` in the same directory. You can also specify it manually:
```bash
python score_fcms.py ... --metadata-path /path/to/metadata.json
```

## License

[Specify your license here]

## Contact

For questions or issues, contact [your contact information]
