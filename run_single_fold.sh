python3 main2.py \
    run_name="check_concatenation" \
    base_transformer='roberta-base'\
    seed=42 \
    group='experiments_combining_features' \
    fold_id="3b9047" \
    batch_size=4 \
    max_epochs=5 \
    max_length=512 \
    val_check_intervals=200 \
    patience=10
