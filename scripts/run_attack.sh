DATASET=latent

python run_attack.py --config ./configs/$DATASET.yaml \
    --img_csv ./AdversarialRobustnessCLIP/$DATASET/list_images_$DATASET.csv \
    --dataset_dir ./AdversarialRobustnessCLIP/ \
    --out_dir ./images/$DATASET