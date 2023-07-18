python3 train.py \
experiment.run_name="test_arc3" \
experiment.transformer_name='roberta-base' \
experiment.fold_id="814d6b" \
experiment.in_features=768 \
experiment.batch_size=8 \
experiment.max_epochs=10 \
experiment.max_length=512 \
experiment.gradient_accumulation_steps=1 \
experiment.freeze_backbone=False