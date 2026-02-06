"""
Example usage of the FCM scoring utility.

This script demonstrates:
1. Using the Python API directly
2. Scoring an FCM with ground truth
3. Customizing the scoring parameters
"""

import os
import sys

# Add parent directory to path to import score_fcms
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from score_fcms import score_fcm


def example_basic():
    """Basic usage with default parameters."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Scoring")
    print("=" * 70)
    
    gt_path = os.path.join(os.path.dirname(__file__), "example_ground_truth.csv")
    gen_path = os.path.join(os.path.dirname(__file__), "example_generated_fcm.json")
    
    results = score_fcm(
        gt_csv_path=gt_path,
        gen_json_path=gen_path,
        verbose=True
    )
    
    print("\nResults DataFrame:")
    print(results)
    print()
    return results


def example_with_metadata():
    """Scoring with metadata for enriched semantic matching."""
    print("=" * 70)
    print("EXAMPLE 2: Scoring with Metadata")
    print("=" * 70)
    
    gt_path = os.path.join(os.path.dirname(__file__), "example_ground_truth.csv")
    gen_path = os.path.join(os.path.dirname(__file__), "example_generated_fcm.json")
    metadata_path = os.path.join(os.path.dirname(__file__), "example_metadata.json")
    
    results = score_fcm(
        gt_csv_path=gt_path,
        gen_json_path=gen_path,
        metadata_json_path=metadata_path,
        verbose=True
    )
    
    print("\nResults DataFrame:")
    print(results)
    print()
    return results


def example_custom_parameters():
    """Scoring with custom parameters."""
    print("=" * 70)
    print("EXAMPLE 3: Custom Parameters")
    print("=" * 70)
    
    gt_path = os.path.join(os.path.dirname(__file__), "example_ground_truth.csv")
    gen_path = os.path.join(os.path.dirname(__file__), "example_generated_fcm.json")
    
    # Try different thresholds
    for threshold in [0.5, 0.7, 0.9]:
        print(f"\n--- Threshold: {threshold} ---")
        results = score_fcm(
            gt_csv_path=gt_path,
            gen_json_path=gen_path,
            threshold=threshold,
            verbose=False
        )
        print(f"F1 Score: {results['F1'].iloc[0]:.4f}")
        print(f"Jaccard:  {results['Jaccard'].iloc[0]:.4f}")
        print(f"TP: {int(results['TP'].iloc[0])}, FP: {int(results['FP'].iloc[0])}, "
              f"FN: {int(results['FN'].iloc[0])}, PP: {int(results['PP'].iloc[0])}")


def example_custom_output():
    """Scoring with custom output directory."""
    print("=" * 70)
    print("EXAMPLE 4: Custom Output Directory")
    print("=" * 70)
    
    gt_path = os.path.join(os.path.dirname(__file__), "example_ground_truth.csv")
    gen_path = os.path.join(os.path.dirname(__file__), "example_generated_fcm.json")
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    
    results = score_fcm(
        gt_csv_path=gt_path,
        gen_json_path=gen_path,
        output_dir=output_dir,
        verbose=True
    )
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    print("\nFCM SCORING EXAMPLES")
    print("=" * 70)
    
    # Run examples
    results1 = example_basic()
    results2 = example_with_metadata()
    example_custom_parameters()
    example_custom_output()
    
    print("\n" + "=" * 70)
    print("Examples completed! Check the output files for saved results.")
    print("=" * 70)
