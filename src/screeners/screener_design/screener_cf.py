from src.screeners.screener import Screener
# from src.embedders.esm2 import EmbedderESM2
from src.utils import calculate_metrics, random_forest_feature_importance_plot

from sklearn.ensemble import RandomForestClassifier as rfc
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import joblib
import yaml

class PeptideScreenerCF(Screener):

    def __init__(self, seq_header:str='sequence', label_header:str='label', embedder_key='CF'):
        super().__init__(seq_header=seq_header, label_header=label_header)

        self.output_dir = None
        self.training_data_dir = None
        self.feature_folder = None
        self.trained = False

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
            "embedder": 'CUSTOM FEATURES',
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

        self.train_eval(train_df, val_df)
        self.gen_config()

    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        """
        returns train, test, val embeddings in this order
        """

        train_data = pd.read_csv(self.feature_folder / 'features_train_df.csv', index_col=False)
        val_data = pd.read_csv(self.feature_folder / 'features_val_df.csv', index_col=False)

        return train_data, val_data
    
    def validate(self, train_data:pd.DataFrame, val_data:pd.DataFrame, outdir:Path):

        X_train,y_train = train_data.drop(columns=[self.seq_header,self.label_header]), train_data[self.label_header]
        X_val,y_val = val_data.drop(columns=[self.seq_header,self.label_header]), val_data[self.label_header]

        prob_train = self.clf.predict_proba(X_train)[:,1]
        prob_test = self.clf.predict_proba(X_val)[:,1]

        calculate_metrics(y_train,prob_train,outdir,title='train')
        calculate_metrics(y_val,prob_test,outdir,title='validation', print_acc=True)

    def train_eval(self, df_train:pd.DataFrame, df_val:pd.DataFrame):

        df_train = self.prepare_dataframe(df_train)
        df_val = self.prepare_dataframe(df_val)

        print(f'training data: {len(df_train)}')
        print(f'validation data: {len(df_val)}')

        print('\n--- TRAINING CLASSIFIER ---\n')

        X,y = df_train.drop(columns=[self.seq_header,self.label_header]), df_train[self.label_header]

        self.clf.fit(X,y)

        print('\n--- RUNNING EVALUATION AND SAVING RESULTS ---\n')
        self.validate(
            train_data=df_train,
            val_data=df_val,
            outdir=self.output_dir
        )

        random_forest_feature_importance_plot(
            model=self.clf,
            feature_names=df_train.drop(columns=[self.seq_header,self.label_header]).columns,
            outdir=self.output_dir
        )

        joblib.dump(self.clf, self.output_dir / 'clf.pkl')
    

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        categorical_threshold: int = 25,
        convert_bool_to_int: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Converts DataFrame columns to numeric or categorical types suitable for ML training.
        
        Strategy:
        - bool            → int (0/1)           (if convert_bool_to_int=True)
        - object/string   → category (if low cardinality) or leave as object
        - int with few unique values → category
        - float/int       → numeric (pd.to_numeric with errors='coerce')
        - columns with too many NaNs after conversion → warn
        
        Parameters:
            df                  : input DataFrame
            exclude_columns     : list of column names to skip (default: ['sequence', 'label'])
            categorical_threshold : max unique values to consider as category
            convert_bool_to_int : convert True/False to 1/0
            verbose             : print what was changed
            
        Returns:
            DataFrame with converted dtypes (copy)
        """
        exclude_columns = [self.seq_header, self.label_header]
        
        df = df.copy()
        changed = []
        
        for col in df.columns:
            if col in exclude_columns:
                if verbose:
                    print(f"Skipping excluded column: {col}")
                continue
                
            original_dtype = df[col].dtype
            unique_count = df[col].nunique(dropna=False)
            
            # 1. Boolean columns
            if pd.api.types.is_bool_dtype(df[col]) or original_dtype == bool:
                if convert_bool_to_int:
                    df[col] = df[col].astype(int)
                    changed.append(f"{col}: bool → int (0/1)")
                continue
                
            # 2. Object / string columns
            if pd.api.types.is_object_dtype(df[col]) or original_dtype == object:
                # Very low cardinality → category
                if unique_count <= categorical_threshold and unique_count > 1:
                    df[col] = df[col].astype('category')
                    changed.append(f"{col}: object → category ({unique_count} levels)")
                # Single value or almost empty → leave as category anyway
                elif unique_count <= 2:
                    df[col] = df[col].astype('category')
                    changed.append(f"{col}: object → category (very low cardinality)")
                # else: leave as object (text features for later encoding)
                continue
                
            # 3. Numeric-looking columns stored as object/string
            if original_dtype == object:
                numeric_converted = pd.to_numeric(df[col], errors='coerce')
                na_before = df[col].isna().sum()
                na_after = numeric_converted.isna().sum()
                
                if na_after - na_before < len(df) * 0.3:  # less than 30% new NaNs
                    df[col] = numeric_converted
                    new_dtype = df[col].dtype
                    changed.append(f"{col}: object → {new_dtype} (coerced)")
                    continue
            
            # 4. Integer columns with low cardinality → probably categorical
            if pd.api.types.is_integer_dtype(df[col]) and unique_count <= categorical_threshold:
                df[col] = df[col].astype('category')
                changed.append(f"{col}: int → category ({unique_count} levels)")
                continue
                
            # 5. Try to make sure float/int columns are proper numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any() and verbose:
                    print(f"Warning: {col} contains NaN after numeric conversion")
        
        return df