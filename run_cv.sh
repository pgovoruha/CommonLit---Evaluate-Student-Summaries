for fold in "3b9047" "39c16e" "814d6b" "ebad26"; do
    python3 train.py \
    experiment.run_name="roberta_base_v1_$fold" \
    experiment.transformer_name='roberta-base' \
    experiment.fold_id=$fold \
    experiment.in_features=768 \
    experiment.batch_size=8 \
    experiment.max_epochs=10 \
    experiment.max_length=512 \
    experiment.logger=wandb
done


