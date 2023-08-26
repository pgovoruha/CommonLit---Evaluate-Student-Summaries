for fold in "3b9047" "39c16e" "814d6b" "ebad26"; do
    python3 main.py \
    run_name="deberta_large_$fold" \
    base_transformer='microsoft/deberta-v3-large'\
    seed=42 \
    group='deberta_large_freeze_embeddings' \
    fold_id=$fold \
    batch_size=4 \
    max_epochs=5 \
    max_length=512 \
    val_check_intervals=200 \
    patience=10

    sleep 5
done


