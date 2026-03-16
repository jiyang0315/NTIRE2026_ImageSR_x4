CUDA_VISIBLE_DEVICES=0 python eval.py \
--output_folder "/home/jiyang/jiyang/Projects/NTIRE2026_ImageSR_x4/output/best_ablation_dt_static_30000_g6_c0p9_s50_adain_2" \
--target_folder "/home/jiyang/jiyang/Projects/SeeSR/preset/datasets/test_datasets/DIV2K/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \