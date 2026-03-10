from src.screeners.screener import Screener# from src.embedders.esm2 import EmbedderESM2
from src.utils import calculate_metrics, get_embedder

from sklearn.ensemble import RandomForestClassifier as rfc
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import joblib
import yaml

class PeptideScreenerPLM(Screener):

    def __init__(self, seq_header:str='sequence', label_header:str='label', embedder_key='ESM2'):
        super().__init__(seq_header=seq_header, label_header=label_header)

        self.output_dir = None
        self.training_data_dir = None
        self.embeddings_folder = None
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

        self.prepare_embeddings(train_df, val_df)
        self.train_eval()
        self.gen_config()
    
    def prepare_embeddings(self, df_train:pd.DataFrame, df_val:pd.DataFrame):

        print('\n generating embeddings')
        
        train_embeddings = self.embedder.get_embeddings(df_train[self.seq_header].to_list())
        val_embeddings = self.embedder.get_embeddings(df_val[self.seq_header].to_list())

        train_labels = df_train[self.label_header].to_numpy()
        val_labels = df_val[self.label_header].to_numpy()

        output_folder = self.output_dir / 'embeddings'
        output_folder.mkdir(parents=True, exist_ok=True)

        self.embeddings_folder = output_folder
        np.savez(output_folder / 'train.npz',train_embeddings,train_labels)
        np.savez(output_folder / 'val.npz',val_embeddings,val_labels)

    def load_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        returns train, test, val embeddings in this order
        """

        train_data = np.load(self.embeddings_folder / 'train.npz')
        val_data = np.load(self.embeddings_folder / 'val.npz')

        return train_data, val_data
    
    def validate(self, train_data, val_data, outdir:Path):

        """
        - Compute performance on train, validation and test sets
        - visualize results 
        """

        X_train, y_train = train_data['arr_0'], train_data['arr_1']
        X_val, y_val = val_data['arr_0'], val_data['arr_1']

        prob_train = self.clf.predict_proba(X_train)[:,1]
        prob_val = self.clf.predict_proba(X_val)[:,1]

        calculate_metrics(y_train,prob_train,outdir,title='train')
        calculate_metrics(y_val,prob_val,outdir,title='validation', print_acc=True)

    def train_eval(self):

        train_data, val_data = self.load_embeddings()

        X_train, y_train = train_data['arr_0'], train_data['arr_1']
        X_val, y_val = val_data['arr_0'], val_data['arr_1']

        print(f'training data: {len(X_train)}')
        print(f'validation data: {len(X_val)}')

        print('\n--- TRAINING CLASSIFIER ---\n')
        self.clf = self.clf.fit(X_train, y_train)

        print('\n--- RUNNING EVALUATION AND SAVING RESULTS ---\n')
        self.validate(
            train_data=train_data,
            val_data=val_data,
            outdir=self.output_dir
        )

        joblib.dump(self.clf, self.output_dir / 'clf.pkl')
    

    