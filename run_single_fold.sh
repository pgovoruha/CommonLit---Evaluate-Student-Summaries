python3 main.py \
    run_name="deberta_base_fold_" \
    base_transformer='microsoft/deberta-v3-base'\
    seed=42 \
    group='deberta-base' \
    fold_id="ebad26" \
    batch_size=4 \
    max_epochs=5 \
    max_length=512 \
    val_check_intervals=200 \
    patience=10