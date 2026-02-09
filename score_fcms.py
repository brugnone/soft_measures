import pandas as pd
import json
import numpy as np
import ast
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
import os
import argparse
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple, Optional


class ScoreCalculator:
    cache = {}

    def __init__(self, threshold, model_name, data, tp_scale=1, pp_scale=0.6, seed=42):
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model_name = model_name
        self.data = data
        self.threshold = threshold
        self.tp_scale = tp_scale
        self.pp_scale = pp_scale

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32  # <- force fp32
        )
        self.model.eval()  # <- disable dropout
        self.task_instruction = "Retrieve clusters that are semantically similar."

        try:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            self.model = self.model.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e) or "MPS backend out of memory" in str(e):
                print("GPU memory insufficient, falling back to CPU...")
                self.device = "cpu"
                self.model = self.model.to(self.device)
            else:
                raise e

    def mean_pool(self, last_hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        summed = (last_hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _sanitize(self, x):
        """Sanitize input while preserving alignment - don't drop entries."""
        s = "" if x is None else str(x).strip()
        return s if s else "<EMPTY>"

    def embed_and_score(self, queries, documents, batch_size=4):
        """
        Compute similarity matrix between queries and documents.

        Returns:
            torch.Tensor: Similarity matrix of shape (len(queries), len(documents))
                         where result[i, j] is similarity between queries[i] and documents[j]
        """
        # Preserve alignment by sanitizing but not filtering out entries
        clean_queries = [self._sanitize(q) for q in queries]
        clean_documents = [self._sanitize(d) for d in documents]

        if not clean_queries or not clean_documents:
            return torch.zeros((len(queries), len(documents)))

        text_prompt = lambda t: f"Instruct: {self.task_instruction}\nQuery: {t}"
        query_texts = [text_prompt(q) for q in clean_queries]
        document_texts = clean_documents

        query_embeddings = []
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch)
            batch_embeddings = self.mean_pool(outputs.last_hidden_state, batch['attention_mask'])
            query_embeddings.append(batch_embeddings)

            del batch, outputs, batch_embeddings
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        document_embeddings = []
        for i in range(0, len(document_texts), batch_size):
            batch_texts = document_texts[i:i + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch)
            batch_embeddings = self.mean_pool(outputs.last_hidden_state, batch['attention_mask'])
            document_embeddings.append(batch_embeddings)

            del batch, outputs, batch_embeddings
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not query_embeddings or not document_embeddings:
            return torch.zeros((len(queries), len(documents)))

        query_embeddings = torch.cat(query_embeddings, dim=0)
        document_embeddings = torch.cat(document_embeddings, dim=0)

        # Robust normalization (no NaNs)
        query_embeddings = torch.nan_to_num(query_embeddings)
        document_embeddings = torch.nan_to_num(document_embeddings)

        # safe_l2_norm
        def safe_l2_norm(x):
            return x / torch.clamp(torch.norm(x, dim=1, keepdim=True), min=1e-6)

        query_embeddings = safe_l2_norm(query_embeddings)
        document_embeddings = safe_l2_norm(document_embeddings)

        similarity_matrix = torch.zeros(len(clean_queries), len(clean_documents), device=self.device)

        for i in range(0, len(clean_queries), batch_size):
            query_batch = query_embeddings[i:i + batch_size]
            for j in range(0, len(clean_documents), batch_size):
                doc_batch = document_embeddings[j:j + batch_size]
                batch_similarity = query_batch @ doc_batch.T
                similarity_matrix[i:i + batch_size, j:j + batch_size] = batch_similarity

                del batch_similarity
                if torch.cuda.is_available() and self.device == "cuda":
                    torch.cuda.empty_cache()

        return similarity_matrix

    def calculate_f1_score(self, tp, fp, fn, pp=None):
        if pp is None:
            denom = 2 * tp + fp + fn
            return (2 * tp) / denom if denom else 0.0
        denom = 2 * tp + fp + fn + pp
        return (2 * tp + pp) / denom if denom else 0.0

    def calculate_jaccard_score(self, tp, fp, fn, pp=None):
        if pp is None:
            denom = tp + fp + fn
            return tp / denom if denom else 0.0
        denom = tp + 0.75 * pp + fp + fn
        return (tp + 0.75 * pp) / denom if denom else 0.0

    def convert_matrix(self, df_matrix):
        values = df_matrix.values.copy()
        columns = df_matrix.columns
        index = df_matrix.index

        # Extract all non-zero entries (including negative values) as edges
        values[values == '0'] = 0
        row_idx, col_idx = np.nonzero(values)

        sources_list = [index[r] for r in row_idx]  # rows → sources
        targets_list = [columns[c] for c in col_idx]  # cols → targets
        
        # Convert edge values to float, treating text-based values as +1.0
        values_list = []
        for r, c in zip(row_idx, col_idx):
            val = values[r, c]
            try:
                # Try to convert to float
                numeric_val = float(val)
                values_list.append(numeric_val)
            except (ValueError, TypeError):
                # If it's text (like "Neutral"), treat as +1.0
                values_list.append(1.0)

        return sources_list, targets_list, values_list

    def calculate_scores(self, fcm1_matrix, fcm2_matrix):
        # Store matrix shapes for accurate node counts (includes isolated nodes)
        self.fcm1_total_nodes = len(fcm1_matrix.index)
        self.fcm2_total_nodes = len(fcm2_matrix.index)
        
        self.fcm1_nodes_src, self.fcm1_nodes_tgt, self.fcm1_edge_dir = self.convert_matrix(fcm1_matrix)
        self.fcm2_nodes_src, self.fcm2_nodes_tgt, self.fcm2_edge_dir = self.convert_matrix(fcm2_matrix)

        # Create a more specific cache key that includes metadata state
        import hashlib, json
        cache_content = (
            tuple(self.fcm2_nodes_src), tuple(self.fcm1_nodes_src),
            tuple(self.fcm2_nodes_tgt), tuple(self.fcm1_nodes_tgt)
        )
        cache_hash = hashlib.md5(str(cache_content).encode()).hexdigest()[:8]
        cache_key = f"{self.data}_{self.model_name}_{cache_hash}"

        if cache_key not in type(self).cache:
            type(self).cache[cache_key] = {}

        if 'src' not in type(self).cache[cache_key]:
            # embed_and_score(queries, documents) returns (queries × documents)
            # Call as (fcm2, fcm1) to get (fcm2 × fcm1) similarity matrix
            src_scores = self.embed_and_score(self.fcm2_nodes_src, self.fcm1_nodes_src, getattr(self, 'batch_size', 2))
            type(self).cache[cache_key]['src'] = src_scores.detach().cpu().numpy()

        if 'tgt' not in type(self).cache[cache_key]:
            # embed_and_score(queries, documents) returns (queries × documents)
            # Call as (fcm2, fcm1) to get (fcm2 × fcm1) similarity matrix
            tgt_scores = self.embed_and_score(self.fcm2_nodes_tgt, self.fcm1_nodes_tgt, getattr(self, 'batch_size', 2))
            type(self).cache[cache_key]['tgt'] = tgt_scores.detach().cpu().numpy()

        all_scores_src = np.array(type(self).cache[cache_key]['src'])
        all_scores_tgt = np.array(type(self).cache[cache_key]['tgt'])

        # Handle empty-edge cases explicitly
        if len(self.fcm2_edge_dir) == 0 or len(self.fcm1_edge_dir) == 0:
            self.TP = self.PP = 0
            self.FP = len(self.fcm2_edge_dir)
            self.FN = len(self.fcm1_edge_dir)

            print("TP, PP, FP, FN:", self.TP, self.PP, self.FP, self.FN)

            TP_scaled = self.TP * self.tp_scale
            PP_scaled = self.PP * self.pp_scale

            F1 = self.calculate_f1_score(TP_scaled, self.FP, self.FN, PP_scaled)
            jaccard = self.calculate_jaccard_score(self.TP, self.FP, self.FN, self.PP)

            model_score = pd.DataFrame({
                'Model': [self.model_name],
                'data': [self.data],
                'F1': [F1],
                'Jaccard': [jaccard],
                'TP': [self.TP],
                'PP': [self.PP],
                'FP': [self.FP],
                'FN': [self.FN],
                'threshold': [self.threshold],
                'tp_scale': [self.tp_scale],
                'pp_scale': [self.pp_scale],
                'fcm1_nodes': [self.fcm1_total_nodes],
                'fcm1_edges': [len(self.fcm1_edge_dir)],
                'fcm2_nodes': [self.fcm2_total_nodes],
                'fcm2_edges': [len(self.fcm2_edge_dir)]
            })

            self.scores_df = model_score
            return model_score

        # Build masks and scores for bipartite matching
        mask = (all_scores_src >= self.threshold) & (all_scores_tgt >= self.threshold)
        combined = (all_scores_src + all_scores_tgt) / 2.0  # fcm2 × fcm1

        # 1-to-1 assignment via Hungarian algorithm with TP tie-breaking bias
        LARGE = 1e6
        sign_mismatch = (np.sign(self.fcm2_edge_dir)[:, None] != np.sign(self.fcm1_edge_dir)[None, :])
        cost = np.where(mask, -combined + 1e-6 * sign_mismatch, LARGE)  # maximize combined, bias toward TP
        row_ind, col_ind = linear_sum_assignment(cost)

        # Keep only valid pairs (passed thresholds)
        matched_pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind) if mask[int(i), int(j)]]

        # Classify matches
        TP = 0
        PP = 0
        for i, j in matched_pairs:
            fcm2_sign = np.sign(self.fcm2_edge_dir[i])
            fcm1_sign = np.sign(self.fcm1_edge_dir[j])
            if fcm2_sign == fcm1_sign:
                TP += 1
            else:
                PP += 1

        matched_fcm2 = {i for i, _ in matched_pairs}
        matched_fcm1 = {j for _, j in matched_pairs}

        FP = len(self.fcm2_edge_dir) - len(matched_fcm2)
        FN = len(self.fcm1_edge_dir) - len(matched_fcm1)

        self.TP, self.PP, self.FP, self.FN = TP, PP, FP, FN

        # Sanity assertions to verify identity constraints
        assert self.TP + self.PP + self.FP == len(self.fcm2_edge_dir), \
            f"FCM2 edge identity violated: {self.TP + self.PP + self.FP} != {len(self.fcm2_edge_dir)}"
        assert self.TP + self.PP + self.FN == len(self.fcm1_edge_dir), \
            f"FCM1 edge identity violated: {self.TP + self.PP + self.FN} != {len(self.fcm1_edge_dir)}"

        # Identity constraints are now guaranteed by construction:
        # TP + PP + FN = len(fcm1_edge_dir)
        # TP + PP + FP = len(fcm2_edge_dir)

        print("TP, PP, FP, FN:", self.TP, self.PP, self.FP, self.FN)

        TP_scaled = self.TP * self.tp_scale
        PP_scaled = self.PP * self.pp_scale

        F1 = self.calculate_f1_score(TP_scaled, self.FP, self.FN, PP_scaled)
        jaccard = self.calculate_jaccard_score(self.TP, self.FP, self.FN, self.PP)

        model_score = pd.DataFrame({
            'Model': [self.model_name],
            'data': [self.data],
            'F1': [F1],
            'Jaccard': [jaccard],
            'TP': [self.TP],
            'PP': [self.PP],
            'FP': [self.FP],
            'FN': [self.FN],
            'threshold': [self.threshold],
            'tp_scale': [self.tp_scale],
            'pp_scale': [self.pp_scale],
            'fcm1_nodes': [self.fcm1_total_nodes],
            'fcm1_edges': [len(self.fcm1_edge_dir)],
            'fcm2_nodes': [self.fcm2_total_nodes],
            'fcm2_edges': [len(self.fcm2_edge_dir)]
        })

        self.scores_df = model_score

        return model_score


def load_matrix_from_file(filepath: str) -> pd.DataFrame:
    """
    Load a matrix from CSV or JSON file.
    
    Args:
        filepath: Path to CSV or JSON file
        
    Returns:
        pd.DataFrame: Adjacency matrix with node names as index and columns
    """
    filepath = str(filepath).lower()
    
    if filepath.endswith('.csv'):
        matrix = pd.read_csv(filepath, index_col=0)
        
        # Drop any unnamed columns (often created by trailing commas)
        unnamed_cols = [col for col in matrix.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            matrix = matrix.drop(columns=unnamed_cols)
        
        # Convert empty strings and string '0' to integer 0
        matrix = matrix.replace('', 0)
        matrix = matrix.replace('""', 0)
        matrix = matrix.replace('0', 0)  # Convert string '0' to integer 0
        
        # Don't force conversion to numeric - let convert_matrix handle text values
        # This preserves values like "Neutral" which will be converted to 1.0 later
        return matrix
    
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        return json_to_matrix(json_data)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .csv or .json")


def json_to_matrix(json_data: Dict) -> pd.DataFrame:
    """Convert FCM JSON to adjacency matrix."""
    edges = []
    for e in json_data.get("edges", []):
        s = str(e["source"])
        t = str(e["target"])
        w = float(e.get("weight", 0.0))
        edges.append((s, t, w))

    # Build adjacency with correct orientation
    if not edges:
        nodes = []
    else:
        nodes = set([s for s, _, _ in edges] + [t for _, t, _ in edges])
    nodes = sorted(nodes) or ["empty_graph"]
    mat = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for s, t, w in edges:
        mat.loc[s, t] = w
    print(f"Created evaluation matrix with {len(edges)} edges and {len(nodes)} nodes")
    return mat


def matrix_to_json(matrix: pd.DataFrame) -> Dict:
    """Convert adjacency matrix to FCM JSON edges format."""
    edges = []
    for src in matrix.index:
        for tgt in matrix.columns:
            weight = float(matrix.loc[src, tgt])
            if weight != 0:
                edges.append({
                    "source": str(src),
                    "target": str(tgt),
                    "weight": weight,
                    "type": "inter_cluster"
                })
    return {"edges": edges}


def load_fcm_data(fcm1_path: str, fcm2_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load two FCM files (CSV or JSON) and return as matrices.
    
    Args:
        fcm1_path: Path to first FCM (CSV or JSON)
        fcm2_path: Path to second FCM (CSV or JSON)
        
    Returns:
        Tuple of (fcm1_matrix, fcm2_matrix)
    """
    print(f"Loading FCM data from {fcm1_path} and {fcm2_path}...")
    fcm1_matrix = load_matrix_from_file(fcm1_path)
    fcm2_matrix = load_matrix_from_file(fcm2_path)
    return fcm1_matrix, fcm2_matrix


def score_fcm(
    fcm1_path: str,
    fcm2_path: str,
    output_dir: Optional[str] = None,
    output_format: str = "csv",
    threshold: float = 0.6,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    tp_scale: float = 1.0,
    pp_scale: float = 0.6,
    batch_size: int = 2,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Score two FCMs against each other.

    Args:
        fcm1_path: Path to first FCM (CSV or JSON)
        fcm2_path: Path to second FCM (CSV or JSON)
        output_dir: Directory to save results. If None, uses fcm2_path directory
        output_format: Output format - 'csv', 'json', or 'both' (default: csv)
        threshold: Similarity threshold for matching (default: 0.6)
        model_name: Embedding model to use (default: Qwen/Qwen3-Embedding-0.6B)
        tp_scale: Scale factor for true positives (default: 1.0)
        pp_scale: Scale factor for partial positives (default: 0.6)
        batch_size: Batch size for processing (lower = less VRAM)
        seed: Random seed for reproducibility
        verbose: Print detailed information

    Returns:
        pd.DataFrame: Results with F1, Jaccard, TP, FP, FN, PP scores and metrics
    """
    # Validate inputs
    if not os.path.exists(fcm1_path):
        raise FileNotFoundError(f"FCM1 file not found: {fcm1_path}")
    if not os.path.exists(fcm2_path):
        raise FileNotFoundError(f"FCM2 file not found: {fcm2_path}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(fcm2_path)
    # If dirname is empty (relative path with no directory), use current directory
    if output_dir == "":
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    data_name = os.path.splitext(os.path.basename(fcm2_path))[0]

    # Load data
    if verbose:
        print(f"Loading FCM data from {fcm1_path} and {fcm2_path}...")
    fcm1_matrix, fcm2_matrix = load_fcm_data(fcm1_path, fcm2_path)

    if verbose:
        print(f"FCM1 matrix shape: {fcm1_matrix.shape}")
        print(f"FCM2 matrix shape: {fcm2_matrix.shape}")

    # Create scorer
    if verbose:
        print("Initializing scorer...")
    scorer = ScoreCalculator(
        threshold=threshold,
        model_name=model_name,
        data=data_name,
        tp_scale=tp_scale,
        pp_scale=pp_scale,
        seed=seed
    )
    scorer.batch_size = batch_size

    # Calculate scores
    if verbose:
        print("Computing embeddings and calculating scores...")
    result = scorer.calculate_scores(fcm1_matrix, fcm2_matrix)

    if verbose:
        print("\n" + "=" * 60)
        print("FCM SCORING RESULTS")
        print("=" * 60)
        print(f"Dataset: {data_name}")
        print(f"Model: {model_name}")
        print(f"Threshold: {threshold}")
        print(f"\nScores:")
        print(f"  F1 Score:      {result['F1'].iloc[0]:.4f}")
        print(f"  Jaccard Score: {result['Jaccard'].iloc[0]:.4f}")
        print(f"\nEdge Matching:")
        print(f"  True Positives:    {scorer.TP}")
        print(f"  Partial Positives: {scorer.PP}")
        print(f"  False Positives:   {scorer.FP}")
        print(f"  False Negatives:   {scorer.FN}")
        print(f"\nGraph Statistics:")
        print(f"  FCM1 Nodes:  {int(result['fcm1_nodes'].iloc[0])}")
        print(f"  FCM1 Edges:  {int(result['fcm1_edges'].iloc[0])}")
        print(f"  FCM2 Nodes:  {int(result['fcm2_nodes'].iloc[0])}")
        print(f"  FCM2 Edges:  {int(result['fcm2_edges'].iloc[0])}")
        print("=" * 60)

    # Save results
    if output_format.lower() in ['csv', 'both']:
        output_file = os.path.join(output_dir, f"{data_name}_scoring_results.csv")
        result.to_csv(output_file, index=False)
        if verbose:
            print(f"Results saved to: {output_file}")
    
    if output_format.lower() in ['json', 'both']:
        output_file = os.path.join(output_dir, f"{data_name}_scoring_results.json")
        result.to_json(output_file, orient='records', indent=2)
        if verbose:
            print(f"Results saved to: {output_file}")

    if verbose:
        print()

    return result


def main():
    """CLI interface for scoring FCM."""
    parser = argparse.ArgumentParser(description='Score two FCMs against each other')
    parser.add_argument('fcm1_path', help='Path to first FCM (CSV or JSON)')
    parser.add_argument('fcm2_path', help='Path to second FCM (CSV or JSON)')
    parser.add_argument('--output-dir', help='Output directory (default: fcm2_path directory)')
    parser.add_argument('--output-format', choices=['csv', 'json', 'both'], default='csv',
                        help='Output format (default: csv)')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold (default: 0.6)')
    parser.add_argument('--model-name', default='Qwen/Qwen3-Embedding-0.6B', help='Model name for scoring')
    parser.add_argument('--tp-scale', type=float, default=1.0, help='True positive scale (default: 1.0)')
    parser.add_argument('--pp-scale', type=float, default=1.1, help='Partial positive scale (default: 1.1)')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    score_fcm(
        fcm1_path=args.fcm1_path,
        fcm2_path=args.fcm2_path,
        output_dir=args.output_dir,
        output_format=args.output_format,
        threshold=args.threshold,
        model_name=args.model_name,
        tp_scale=args.tp_scale,
        pp_scale=args.pp_scale,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()