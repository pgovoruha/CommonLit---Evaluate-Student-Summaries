for fold in "3b9047" "39c16e" "814d6b" "ebad26"; do
    python3 main.py \
    run_name="roberta_base_conc_pool_$fold" \
    base_transformer='roberta-base' \
    fold_id=$fold \
    batch_size=8 \
    max_epochs=10 \
    max_length=512 \
    logger=wandb

    sleep 5
done


