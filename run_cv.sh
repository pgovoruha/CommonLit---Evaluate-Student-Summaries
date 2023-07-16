python3 train.py \
experiment.run_name="roberta_base_v1_fold_0" \
experiment.transformer_name='roberta-base' \
experiment.in_features=768 \
experiment.batch_size=8 \
experiment.max_epochs=5 \
experiment.max_length=512 \
experiment.train_path="data/Egyptian Social Structure/train.csv" \
experiment.val_path="data/Egyptian Social Structure/test.csv" \
experiment.test_path="data/Egyptian Social Structure/test.csv"

python3 train.py \
experiment.run_name="roberta_base_v1_fold_1" \
experiment.transformer_name='roberta-base' \
experiment.in_features=768 \
experiment.batch_size=8 \
experiment.max_epochs=5 \
experiment.max_length=512 \
experiment.train_path="data/Excerpt from The Jungle/train.csv" \
experiment.val_path="data/Excerpt from The Jungle/test.csv" \
experiment.test_path="data/Excerpt from The Jungle/test.csv"

python3 train.py \
experiment.run_name="roberta_base_v1_fold_2" \
experiment.transformer_name='roberta-base' \
experiment.in_features=768 \
experiment.batch_size=8 \
experiment.max_epochs=5 \
experiment.max_length=512 \
experiment.train_path="data/On Tragedy/train.csv" \
experiment.val_path="data/On Tragedy/test.csv" \
experiment.test_path="data/On Tragedy/test.csv"

python3 train.py \
experiment.run_name="roberta_base_v1_fold_3" \
experiment.transformer_name='roberta-base' \
experiment.in_features=768 \
experiment.batch_size=8 \
experiment.max_epochs=5 \
experiment.max_length=512 \
experiment.train_path="data/The Third Wave/train.csv" \
experiment.val_path="data/The Third Wave/test.csv" \
experiment.test_path="data/The Third Wave/test.csv"

