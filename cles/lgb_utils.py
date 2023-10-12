import pandas as pd
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from readability import Readability
import optuna
import lightgbm as lgb
import numpy as np
import random
from sklearn.metrics import mean_squared_error


class Preprocessor:

    def __init__(self):
        self.STOP_WORDS = set(stopwords.words('english'))
        self.spellchecker = SpellChecker()

    def check_is_stop_word(self, word):
        return word in self.STOP_WORDS

    def word_overlap_count(self, row):

        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(self.check_is_stop_word, prompt_words))
            summary_words = list(filter(self.check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))

    def word_overlap_ratio(self, row):
        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(self.check_is_stop_word, prompt_words))
            summary_words = list(filter(self.check_is_stop_word, summary_words))

        try:
            ratio = len(set(prompt_words).intersection(set(summary_words)))/len(set(prompt_words).union(set(summary_words)))
        except Exception as e:
            ratio = 0
        return ratio

    def ngrams(self, token, n):
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)

    def quotes_count(self, row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary) > 0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, words):

        amount_miss = len(list(self.spellchecker.unknown(words)))

        return amount_miss

    def compute_number_of_unique_words(self, words):
        unique_words = set(words)
        return len(unique_words)

    def add_flesch_kinsaid(self, text):
        r = Readability(text)

        try:
            return r.flesch_kincaid().score
        except Exception as e:
            return -100

    def add_flesch(self, text):
        r = Readability(text)
        try:
            return r.flesch().score
        except Exception as e:
            return -100

    def add_gunning_fog(self, text):
        r = Readability(text)
        try:
            return r.gunning_fog().score
        except Exception as e:
            return -100

    def add_coleman_liau(self, text):
        r = Readability(text)
        try:
            return r.coleman_liau().score
        except Exception as e:
            return -100

    def add_dale_chall(self, text):
        r = Readability(text)
        try:
            return r.dale_chall().score
        except Exception as e:
            return -100

    def add_ari(self, text):
        r = Readability(text)
        try:
            return r.ari().score
        except Exception as e:
            return -100

    def add_linsear_write(self, text):
        r = Readability(text)
        try:
            return r.linsear_write().score
        except Exception as e:
            return -100

    def add_smog(self, text):
        r = Readability(text)
        try:
            return r.smog().score
        except Exception as e:
            return -100

    def add_spache(self, text):
        r = Readability(text)
        try:
            return r.spache().score
        except Exception as e:
            return -100

    def run(self,
            prompts: pd.DataFrame,
            summaries: pd.DataFrame,
            ) -> pd.DataFrame:

        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(word_tokenize(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(word_tokenize(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )

        summaries["splling_err_num"] = summaries["summary_tokens"].map(lambda x: self.spelling(x))

        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']
        input_df['word_overlap_count'] = input_df.apply(lambda row: self.word_overlap_count(row), axis=1)
        input_df['word_overlap_ratio'] = input_df.apply(lambda row: self.word_overlap_ratio(row), axis=1)
        input_df['bigram_overlap_count'] = input_df.apply(lambda row: self.ngram_co_occurrence(row, 2), axis=1)
        input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)
        input_df['trigram_overlap_count'] = input_df.apply(lambda row: self.ngram_co_occurrence(row, 3), axis=1)
        input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)
        input_df['quotes_count'] = input_df.apply(lambda row: self.quotes_count(row), axis=1)
        input_df['number_of_unique_words'] = input_df['summary_tokens'].map(
            lambda x: self.compute_number_of_unique_words(x))
        input_df['flesch_kinsaid'] = input_df['text'].map(lambda x: self.add_flesch_kinsaid(x))
        input_df['flesch'] = input_df['text'].map(lambda x: self.add_flesch(x))
        input_df['gunning_fog'] = input_df['text'].map(lambda x: self.add_gunning_fog(x))
        input_df['coleman_liau'] = input_df['text'].map(lambda x: self.add_coleman_liau(x))
        input_df['dale_chall'] = input_df['text'].map(lambda x: self.add_dale_chall(x))
        input_df['ari'] = input_df['text'].map(lambda x: self.add_ari(x))
        input_df['linsear_write'] = input_df['text'].map(lambda x: self.add_linsear_write(x))
        input_df['smog'] = input_df['text'].map(lambda x: self.add_smog(x))
        input_df['spache'] = input_df['text'].map(lambda x: self.add_spache(x))

        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


class LGBRegressor:

    def __init__(self, dataframe, feature_list, target, save_path):

        self.dataframe = dataframe
        self.feature_list = feature_list
        self.target = target
        self.best_params = None
        self.best_score = None
        self.save_path = save_path

    def run_hyperparameters_optimization(self):

        def lgb_objective(trial):
            max_depth = trial.suggest_int('max_depth', 2, 10)
            params = {
                'boosting_type': 'gbdt',
                'random_state': 42,
                'objective': 'regression',
                'metric': 'rmse',
                'n_jobs': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'max_depth': max_depth,
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 2 ** max_depth - 1),
                'verbosity': -1  # Add this line to suppress warnings and info messages

            }

            res = []
            for prompt_id, data in self.dataframe.groupby('prompt_id'):
                x_train = self.dataframe[self.dataframe['prompt_id'] != prompt_id].copy()
                dtrain = lgb.Dataset(data=x_train[self.feature_list], label=x_train[self.target])
                dtest = lgb.Dataset(data=data[self.feature_list], label=data[self.target])

                model = lgb.train(params,
                                  num_boost_round=10000,
                                  valid_names=['valid'],
                                  train_set=dtrain,
                                  valid_sets=dtest,
                                  callbacks=[
                                      lgb.early_stopping(stopping_rounds=200, verbose=False)])

                predictions = model.predict(data[self.feature_list])
                data.loc[:, 'prediction'] = predictions
                res.append(data[[self.target, 'prediction']].copy())

            res_df = pd.concat(res, axis=0, ignore_index=True)
            score = mean_squared_error(res_df[self.target], res_df['prediction'], squared=False)
            return score

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lgb_objective, n_trials=1000)

        best_score = study.best_value
        best_params = {
            'boosting_type': 'gbdt',
            'random_state': 42,
            'objective': 'regression',
            'metric': 'rmse',
            'n_jobs': -1,
            'learning_rate': study.best_params['learning_rate'],
            'max_depth': int(study.best_params['max_depth']),
            'lambda_l1': study.best_params['lambda_l1'],
            'lambda_l2': study.best_params['lambda_l2'],
            'num_leaves': int(study.best_params['num_leaves']),
            'verbosity': -1  # Add this line to suppress warnings and info messages

        }

        self.best_params = best_params
        print('Best score is ', best_score)
        self.best_score = best_score

    def run_feature_selection(self):

        def objective(trial):

            feature_dict = {}
            for feature in self.feature_list:
                feature_dict[feature] = trial.suggest_categorical(feature, ('use', 'drop'))

            updated_feature_list = [k for k, v in feature_dict.items() if v != 'drop']

            res = []
            for prompt_id, data in self.dataframe.groupby('prompt_id'):
                x_train = self.dataframe[self.dataframe['prompt_id'] != prompt_id].copy()
                dtrain = lgb.Dataset(data=x_train[updated_feature_list], label=x_train[self.target])
                dtest = lgb.Dataset(data=data[updated_feature_list], label=data[self.target])

                model = lgb.train(self.best_params,
                                  num_boost_round=10000,
                                  valid_names=['valid'],
                                  train_set=dtrain,
                                  valid_sets=dtest,
                                  callbacks=[
                                      lgb.early_stopping(stopping_rounds=200, verbose=False)])

                predictions = model.predict(data[updated_feature_list])
                data.loc[:, 'prediction'] = predictions
                res.append(data[[self.target, 'prediction']].copy())

            res_df = pd.concat(res, axis=0, ignore_index=True)
            score = mean_squared_error(res_df[self.target], res_df['prediction'], squared=False)
            return score

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=1000)
        best_score = study.best_value
        print('Best score is : ', best_score)
        best_features = [k for k, v in study.best_params.items() if v != 'drop']
        if f'pred_{self.target}' not in best_features:
            best_features.append(f'pred_{self.target}')
        self.feature_list = best_features
        self.best_score = best_score

    def train_and_save(self):

        self.dataframe.loc[:, f'pred_lgb_{self.target}'] = None
        for prompt_id, data in self.dataframe.groupby('prompt_id'):
            x_train = self.dataframe[self.dataframe['prompt_id'] != prompt_id].copy()
            dtrain = lgb.Dataset(data=x_train[self.feature_list], label=x_train[self.target])
            dtest = lgb.Dataset(data=data[self.feature_list], label=data[self.target])

            model = lgb.train(self.best_params,
                              num_boost_round=10000,
                              valid_names=['valid'],
                              train_set=dtrain,
                              valid_sets=dtest,
                              callbacks=[
                                  lgb.early_stopping(stopping_rounds=200, verbose=False)])

            predictions = model.predict(data[self.feature_list])
            self.dataframe.loc[self.dataframe['prompt_id'] == prompt_id, f'pred_lgb_{self.target}'] = predictions
            model.save_model(f'{self.save_path}/lgb_model_{self.target}_{prompt_id}.bst')

        return self.dataframe[['prompt_id', 'student_id', f'pred_lgb_{self.target}']]

    def run(self):
        print('Tuning hyperparameters before applying feature selection')
        self.run_hyperparameters_optimization()
        print('Applying feature selection')
        self.run_feature_selection()
        print('Tuning hyperparameters after feature selection')
        self.run_hyperparameters_optimization()
        print('Training and saving optimized model')
        oof_df = self.train_and_save()
        return oof_df
