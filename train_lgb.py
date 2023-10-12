import os
import optuna
import pandas as pd
import numpy as np
import random
from argparse import ArgumentParser
from cles.lgb_utils import Preprocessor, LGBRegressor
from cles.metrics.metric import mcrmse
import json
import warnings
warnings.filterwarnings('ignore')

processor = Preprocessor()

parser = ArgumentParser()
parser.add_argument('--oof_file_path', dest='oof_file_path',
                    default='oof_predictions/microsoft_deberta_v3_large_mean_pool.csv')
parser.add_argument('--save_path', dest='save_path', default='lgb_models/test')
parser.add_argument('--save_oof_name', dest='save_oof_name', default='lgb_mean_pool_test')

random.seed(42)
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():

    args = parser.parse_args()
    file_path = args.oof_file_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    save_oof_name = args.save_oof_name
    oof_df = pd.read_csv(file_path)
    summaries = pd.read_csv('data_raw/summaries_train.csv')
    prompts = pd.read_csv('data_raw/prompts_train.csv')
    df = processor.run(prompts, summaries)
    oof_df = oof_df.drop(['content', 'wording'], axis=1)
    full_df = df.merge(oof_df, on=['student_id', 'prompt_id'])
    feature_list = full_df.drop(['student_id', 'prompt_id', 'text', 'content', 'wording',
                                       'prompt_question', 'prompt_title',
                                       'prompt_text'
                            ], axis=1).columns.tolist()

    content_regressor = LGBRegressor(dataframe=full_df.copy(),
                                     feature_list=feature_list,
                                     target='content',
                                     save_path=save_path)
    oof_content = content_regressor.run()
    best_features_content = content_regressor.feature_list

    wording_regressor = LGBRegressor(dataframe=full_df.copy(),
                                     feature_list=feature_list,
                                     target='wording',
                                     save_path=save_path)
    oof_wording = wording_regressor.run()
    best_features_wording = wording_regressor.feature_list

    full_df = full_df.merge(oof_content, on=['prompt_id', 'student_id'])
    full_df = full_df.merge(oof_wording, on=['prompt_id', 'student_id'])
    mcrmse_score = mcrmse(full_df[['content', 'wording']].values, full_df[['pred_lgb_content', 'pred_lgb_wording']].values)
    print(f'MCRMSE is {mcrmse_score}')

    with open(f'{save_path}/feature_list.json', 'w') as f:
        json.dump(
            {
                'content_feature_list': best_features_content,
                'wording_feature_list': best_features_wording
            },
            f
        )
    full_df.to_csv(f'oof_predictions/{save_oof_name}.csv', index=False)


if __name__ == '__main__':
    main()












