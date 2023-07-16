python3 train.py \
experiment.run_name="roberta_large_fold0" \
experiment.max_epochs=1 \
experiment.transformer_name='roberta-large' \
experiment.in_features=1024 \
experiment.batch_size=2 \
experiment.max_epochs=1 \
experiment.max_length=512 \
experiment.train_path="data/Egyptian Social Structure/train.csv" \
experiment.val_path="data/Egyptian Social Structure/test.csv" \
experiment.test_path="data/Egyptian Social Structure/test.csv"

python3 train.py \
experiment.run_name="roberta_large_fold1" \
experiment.max_epochs=1 \
experiment.transformer_name='roberta-large' \
experiment.in_features=1024 \
experiment.batch_size=2 \
experiment.max_epochs=1 \
experiment.max_length=512 \
experiment.train_path="data/Excerpt from The Jungle/train.csv" \
experiment.val_path="data/Excerpt from The Jungle/test.csv" \
experiment.test_path="data/Excerpt from The Jungle/test.csv"

python3 train.py \
experiment.run_name="roberta_large_fold2" \
experiment.max_epochs=1 \
experiment.transformer_name='roberta-large' \
experiment.in_features=1024 \
experiment.batch_size=2 \
experiment.max_epochs=1 \
experiment.max_length=512 \
experiment.train_path="data/On Tragedy/train.csv" \
experiment.val_path="data/On Tragedy/test.csv" \
experiment.test_path="data/On Tragedy/test.csv"

python3 train.py \
experiment.run_name="roberta_large_fold3" \
experiment.max_epochs=1 \
experiment.transformer_name='roberta-large' \
experiment.in_features=1024 \
experiment.batch_size=2 \
experiment.max_epochs=1 \
experiment.max_length=512 \
experiment.train_path="data/The Third Wave/train.csv" \
experiment.val_path="data/The Third Wave/test.csv" \
experiment.test_path="data/The Third Wave/test.csv"

