"""Complete OSW dataset only."""

from compare_ai_to_groundtruth import compare_ai_to_groundtruth

def main():
    """Run comparison for OSW only."""
    
    ai_base_dir = r'C:\Users\Nbrug\Desktop\fcm_ai'
    gt_base_dir = r'C:\Users\Nbrug\Desktop\fcm_gt'
    output_dir = r'C:\Users\Nbrug\Desktop\fcm_comparison_results'
    
    print("=" * 70)
    print("COMPLETING OSW DATASET")
    print("=" * 70)
    
    compare_ai_to_groundtruth(
        ai_base_dir=ai_base_dir,
        gt_base_dir=gt_base_dir,
        output_dir=output_dir,
        datasets=['osw'],
        threshold=0.6,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        tp_scale=1.0,
        pp_scale=0.6,
        batch_size=2,
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("OSW COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
