"""Complete the remaining datasets (OSW and Red Snapper) for AI vs GT comparison."""

from compare_ai_to_groundtruth import compare_ai_to_groundtruth

def main():
    """Run comparison for remaining datasets only."""
    
    # Directories
    ai_base_dir = r'C:\Users\Nbrug\Desktop\fcm_ai'
    gt_base_dir = r'C:\Users\Nbrug\Desktop\fcm_gt'
    output_dir = r'C:\Users\Nbrug\Desktop\fcm_comparison_results'
    
    print("=" * 70)
    print("COMPLETING REMAINING DATASETS: OSW AND RED SNAPPER")
    print("=" * 70)
    print(f"\nAI Base Directory: {ai_base_dir}")
    print(f"GT Base Directory: {gt_base_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Run comparison for remaining datasets only
    results = compare_ai_to_groundtruth(
        ai_base_dir=ai_base_dir,
        gt_base_dir=gt_base_dir,
        output_dir=output_dir,
        datasets=['osw', 'red_snapper'],  # Only these two
        threshold=0.6,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        tp_scale=1.0,
        pp_scale=0.6,
        batch_size=2,
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("REMAINING DATASETS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
