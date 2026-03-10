"""
SCREENER BASED ON SOLUBILITY CLASSIFICATION
"""

from src.screeners.screener import Screener
# from src.screeners.solubility.embedder_bert import EmbedderBert as Embedder
from src.feature_generators.PLM.pbert import EmbedderBERT as Embedder

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

class SolubilityScreenerJana(Screener):

    def __init__(self, model_path:Path, device:str=None, seq_header:str = 'sequence', thr:float=0.55):
        
        print(device)
        super().__init__(seq_header=seq_header, device=device)

        self.model = joblib.load(model_path)
        self.threshold = thr

        # Initialize the ProtBERT tokenizer and model
        #self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, use_fast=False)
        self.embedder = Embedder(self.device)

    def run_screening(self, df:pd.DataFrame):

        sequences = self.preprocess_sequences(df[self.seq_header].to_list())

        probabilities = np.array(self.model.predict_proba(sequences)[:, 1], dtype=np.float32)  # Probabilities for the positive class
        predictions = (probabilities > self.threshold).astype(int)
        df['jana_solubility'] = predictions

        return df
    
    def preprocess_sequences(self, sequences) -> np.ndarray:
        """
        - embedd sequences via ESM-2 model
        - batch size is 4 by default ! (len(sequences) has to be at least 4 !)
        """
        print('generating sequence embeddings')
        return self.embedder.get_embeddings(sequences)