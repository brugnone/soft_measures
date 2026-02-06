"""
Example usage of the FCM scoring utility.

This script demonstrates:
1. Using the Python API directly
2. Scoring two FCMs with different formats
3. Customizing the scoring parameters
4. Handling different output formats
"""

import os
import sys

# Add parent directory to path to import score_fcms
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from score_fcms import score_fcm


def example_basic():
    """Basic usage with CSV format."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Scoring (CSV vs CSV)")
    print("=" * 70)
    
    fcm1_path = os.path.join(os.path.dirname(__file__), "fcm1.csv")
    fcm2_path = os.path.join(os.path.dirname(__file__), "fcm2.csv")
    
    results = score_fcm(
        fcm1_path=fcm1_path,
        fcm2_path=fcm2_path,
        verbose=True
    )
    
    print("\nResults DataFrame:")
    print(results)
    print()
    return results


def example_output_formats():
    """Scoring with different output formats."""
    print("=" * 70)
    print("EXAMPLE 2: Different Output Formats")
    print("=" * 70)
    
    fcm1_path = os.path.join(os.path.dirname(__file__), "fcm1.csv")
    fcm2_path = os.path.join(os.path.dirname(__file__), "fcm2.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "results_csv_json")
    
    results = score_fcm(
        fcm1_path=fcm1_path,
        fcm2_path=fcm2_path,
        output_dir=output_dir,
        output_format='both',
        verbose=True
    )
    
    print("\nResults saved in both CSV and JSON formats!")


def example_custom_parameters():
    """Scoring with custom parameters."""
    print("=" * 70)
    print("EXAMPLE 3: Custom Parameters - Threshold Tuning")
    print("=" * 70)
    
    fcm1_path = os.path.join(os.path.dirname(__file__), "fcm1.csv")
    fcm2_path = os.path.join(os.path.dirname(__file__), "fcm2.csv")
    
    # Try different thresholds
    print("\nTesting different thresholds:\n")
    for threshold in [0.5, 0.7, 0.9]:
        print(f"--- Threshold: {threshold} ---")
        results = score_fcm(
            fcm1_path=fcm1_path,
            fcm2_path=fcm2_path,
            threshold=threshold,
            verbose=False
        )
        print(f"F1 Score: {results['F1'].iloc[0]:.4f}")
        print(f"Jaccard:  {results['Jaccard'].iloc[0]:.4f}")
        print(f"Edges - TP: {int(results['TP'].iloc[0])}, FP: {int(results['FP'].iloc[0])}, "
              f"FN: {int(results['FN'].iloc[0])}, PP: {int(results['PP'].iloc[0])}")
        print()


def example_json_to_csv():
    """Score with JSON output format."""
    print("=" * 70)
    print("EXAMPLE 4: JSON Format Output")
    print("=" * 70)
    
    fcm1_path = os.path.join(os.path.dirname(__file__), "fcm1.csv")
    fcm2_path = os.path.join(os.path.dirname(__file__), "fcm2.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "results_json_only")
    
    results = score_fcm(
        fcm1_path=fcm1_path,
        fcm2_path=fcm2_path,
        output_dir=output_dir,
        output_format='json',
        verbose=True
    )
    
    print("\nResults saved in JSON format!")


if __name__ == "__main__":
    print("\nFCM SCORING EXAMPLES")
    print("=" * 70)
    
    # Run examples
    results1 = example_basic()
    example_output_formats()
    example_custom_parameters()
    example_json_to_csv()
    
    print("\n" + "=" * 70)
    print("Examples completed! Check the output files for saved results.")
    print("=" * 70)
