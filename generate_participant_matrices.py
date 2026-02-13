"""
Generate individual adjacency matrices for each participant from FCM extraction data.

This script reads an Excel file containing FCM data with source nodes (From),
target nodes (To), edge weights (relation), and individual participant responses.
It creates a separate adjacency matrix CSV for each participant.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def create_adjacency_matrix_for_participant(df, from_col, to_col, relation_col, participant_col):
    """
    Create an adjacency matrix for a single participant.
    
    Args:
        df: Full dataframe with all edge data
        from_col: Name of source node column
        to_col: Name of target node column
        relation_col: Name of edge weight column
        participant_col: Name of participant column (contains 0/1 for edge inclusion)
    
    Returns:
        pd.DataFrame: Square adjacency matrix
    """
    # Filter to only edges where this participant indicated presence (value = 1)
    participant_edges = df[df[participant_col] == 1].copy()
    
    if len(participant_edges) == 0:
        print(f"  Warning: No edges found for {participant_col}")
        return None
    
    # Get all unique nodes (from both From and To columns)
    all_nodes = sorted(set(participant_edges[from_col].unique()) | 
                      set(participant_edges[to_col].unique()))
    
    # Create empty adjacency matrix
    adj_matrix = pd.DataFrame(0, index=all_nodes, columns=all_nodes)
    
    # Fill in the edge weights
    for _, row in participant_edges.iterrows():
        source = row[from_col]
        target = row[to_col]
        weight = row[relation_col]
        adj_matrix.loc[source, target] = weight
    
    return adj_matrix


def generate_participant_adjacency_matrices(
    excel_path,
    output_dir,
    from_col='From',
    to_col='To ',  # Note the space
    relation_col='relation',
    description_col='description',
    verbose=True
):
    """
    Generate individual adjacency matrix CSVs for each participant.
    
    Args:
        excel_path: Path to Excel file
        output_dir: Directory to save CSV files
        from_col: Column name for source nodes (default: 'From')
        to_col: Column name for target nodes (default: 'To ')
        relation_col: Column name for edge weights (default: 'relation')
        description_col: Column after which participant columns start (default: 'description')
        verbose: Print progress information
    """
    # Load the Excel file
    if verbose:
        print(f"Loading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    
    if verbose:
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Find participant columns (all columns after description_col)
    try:
        desc_idx = list(df.columns).index(description_col)
        participant_cols = list(df.columns[desc_idx + 1:])
    except ValueError:
        print(f"Error: '{description_col}' column not found")
        return
    
    if verbose:
        print(f"\nFound {len(participant_cols)} participant columns:")
        print(f"  {', '.join(participant_cols[:10])}{'...' if len(participant_cols) > 10 else ''}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"\nOutput directory: {output_dir}")
    
    # Generate adjacency matrix for each participant
    successful = 0
    failed = 0
    
    if verbose:
        print(f"\nGenerating adjacency matrices...")
        print("=" * 70)
    
    for i, participant_col in enumerate(participant_cols, 1):
        if verbose:
            print(f"[{i}/{len(participant_cols)}] Processing: {participant_col}")
        
        try:
            # Create adjacency matrix
            adj_matrix = create_adjacency_matrix_for_participant(
                df, from_col, to_col, relation_col, participant_col
            )
            
            if adj_matrix is not None:
                # Save to CSV
                output_file = os.path.join(output_dir, f"{participant_col}.csv")
                adj_matrix.to_csv(output_file)
                
                if verbose:
                    print(f"  ✓ Saved: {participant_col}.csv ({adj_matrix.shape[0]} nodes, "
                          f"{(adj_matrix != 0).sum().sum()} edges)")
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error processing {participant_col}: {e}")
            failed += 1
    
    # Summary
    if verbose:
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Successfully generated: {successful} files")
        if failed > 0:
            print(f"  Failed: {failed} files")
        print(f"\nAll files saved to: {output_dir}")


def main():
    """Main function to process the Red Snapper FCM data."""
    
    # File paths
    excel_path = r'C:\Users\Nbrug\Downloads\All FCM Extraction Paper Data [formatted]-20260213T160502Z-1-001\All FCM Extraction Paper Data [formatted]\Red snapper\full model.xlsx'
    output_dir = r'C:\Users\Nbrug\Desktop\Red_Snapper_gt'
    
    print("=" * 70)
    print("FCM ADJACENCY MATRIX GENERATOR")
    print("=" * 70)
    print(f"\nInput: {excel_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Generate matrices
    generate_participant_adjacency_matrices(
        excel_path=excel_path,
        output_dir=output_dir,
        from_col='From',
        to_col='To ',  # Note: there's a space after 'To'
        relation_col='relation',
        description_col='description',
        verbose=True
    )
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
