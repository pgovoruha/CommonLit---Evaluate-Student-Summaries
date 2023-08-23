python3 main.py \
    run_name="deberta_large_fold_ebad26" \
    base_transformer='microsoft/deberta-v3-large'\
    seed=42 \
    group='deberta-large baseline' \
    fold_id="ebad26" \
    batch_size=4 \
    max_epochs=5 \
    max_length=512 \
    val_check_intervals=200 \
    patience=10
