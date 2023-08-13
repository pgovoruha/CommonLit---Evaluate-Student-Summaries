for fold in "3b9047" "39c16e" "814d6b" "ebad26"; do
    python3 main.py \
    run_name="roberta_base_$fold" \
    base_transformer='roberta-base'\
    seed=42 \
    group='tsry_mcrmse_loss' \
    fold_id=$fold \
    batch_size=4 \
    max_epochs=5 \
    max_length=512 \
    val_check_intervals=200 \
    patience=5

    sleep 5
done


