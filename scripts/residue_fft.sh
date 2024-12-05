DATASET=latent

python residue_fft.py --config ./configs/$DATASET.yaml \
    --img_csv ./AdversarialRobustnessCLIP/$DATASET/list_images_$DATASET.csv \
    --adv_dir ./images/$DATASET \
    --out_dir ./fft_images/$DATASET