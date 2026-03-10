from src.screeners.screener import Screener
# from src.embedders.esm2 import EmbedderESM2
from src.utils import calculate_metrics, get_embedder, random_forest_feature_importance_plot

from sklearn.ensemble import RandomForestClassifier as rfc
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import joblib
import yaml

class PeptideScreenerPCHEM(Screener):

    def __init__(self, seq_header:str='sequence', label_header:str='label', embedder_key='PCHEM'):
        super().__init__(seq_header=seq_header, label_header=label_header)

        self.output_dir = None
        self.training_data_dir = None
        self.feature_folder = None
        self.trained = False

        self.embedder = get_embedder(embedder_key)(
            device=self.device
        )
        self.clf = rfc(
            # prevent overfitting
            n_estimators=50,
            max_depth=20,
            criterion='gini',
            min_samples_leaf=1
        )
    
    def gen_config(self):
        config = {
            "classifier": 'RFR',
            "embedder": self.embedder.name,
        }

        config_path = self.output_dir / "config.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )

    def design_screener(self, train_df:pd.DataFrame, val_df:pd.DataFrame, output_dir:Path):

        self.output_dir = output_dir

        train_df[self.label_header] = train_df[self.label_header].astype('category')
        val_df[self.label_header] = val_df[self.label_header].astype('category')

        self.prepare_features(train_df, val_df)
        self.train_eval()
        self.gen_config()
    
    def prepare_features(self, df_train:pd.DataFrame, df_val:pd.DataFrame):

        print('\n generating embeddings')

        train_df = self.embedder.get_features(df_train[self.seq_header].to_list(),df_train[self.label_header].to_list())
        val_df = self.embedder.get_features(df_val[self.seq_header].to_list(), df_val[self.label_header].to_list())

        output_folder = self.output_dir / 'features'
        output_folder.mkdir(parents=True, exist_ok=True)

        self.feature_folder = output_folder
        train_df.to_csv(output_folder / 'features_train_df.csv', index=False)
        val_df.to_csv(output_folder / 'features_val_df.csv', index=False)

    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        """
        returns train, test, val embeddings in this order
        """

        train_data = pd.read_csv(self.feature_folder / 'features_train_df.csv', index_col=False)
        val_data = pd.read_csv(self.feature_folder / 'features_val_df.csv', index_col=False)

        return train_data, val_data
    
    def validate(self, train_data:pd.DataFrame, val_data:pd.DataFrame, outdir:Path):

        X_train,y_train = train_data.iloc[:,1:-1], train_data.iloc[:,-1]
        X_val,y_val = val_data.iloc[:,1:-1], val_data.iloc[:,-1]

        prob_train = self.clf.predict_proba(X_train)[:,1]
        prob_test = self.clf.predict_proba(X_val)[:,1]

        calculate_metrics(y_train,prob_train,outdir,title='train')
        calculate_metrics(y_val,prob_test,outdir,title='validation', print_acc=True)

    def train_eval(self):

        train_data, val_data = self.load_features()

        print(f'training data: {len(train_data)}')
        print(f'validation data: {len(val_data)}')

        print('\n--- TRAINING CLASSIFIER ---\n')

        X,y = train_data.iloc[:,1:-1], train_data.iloc[:,-1]

        self.clf.fit(X,y)

        print('\n--- RUNNING EVALUATION AND SAVING RESULTS ---\n')
        self.validate(
            train_data=train_data,
            val_data=val_data,
            outdir=self.output_dir
        )

        random_forest_feature_importance_plot(
            model=self.clf,
            feature_names=train_data.iloc[:,1:-1].columns,
            outdir=self.output_dir
        )

        joblib.dump(self.clf, self.output_dir / 'clf.pkl')
    

    