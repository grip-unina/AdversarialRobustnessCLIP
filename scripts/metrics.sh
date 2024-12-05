DATASET=latent

python metrics.py --config ./configs/$DATASET.yaml \
    --img_csv ./AdversarialRobustnessCLIP/$DATASET/list_images_$DATASET.csv \
    --dataset_dir ./AdversarialRobustnessCLIP/ \
    --adv_dir ./images/$DATASET \
    --res_dir ./res_csv/$DATASET