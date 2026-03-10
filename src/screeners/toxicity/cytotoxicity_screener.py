"""
SCREENER BASED ON CYTOTOXICITY CLASSIFICATION
"""

from src.screeners.screener import Screener
# from src.screeners.toxicity.embedder import Embedder

from src.feature_generators.PLM.esm2 import EmbedderESM2 as Embedder
import numpy as np
import pandas as pd

from pathlib import Path
import joblib

class CytotoxicityScreener(Screener):

    def __init__(self, model_path:Path, device=None, seq_header:str='sequence'):

        super().__init__(device, seq_header)

        self.model = joblib.load(model_path)

        print('initializing the embedding model')
        self.embedder = Embedder(self.device)

    def run_screening(self, df:pd.DataFrame):

        """
        run cytotoxicity filtering
        RETURN: pandas datafarme with added columns 'toxicity_prob', 'toxicity_cat and embedded sequences'
        """
        sequences = self.preprocess_sequences(list(df[self.seq_header].to_numpy()))
        probabilities = np.array(self.model.predict_proba(sequences)[:,1], dtype=np.float32)

        df['toxicity_prob'] = probabilities

        low, med = 0.5, 0.8

        df['toxicity_cat'] = np.select(
            [df['toxicity_prob'] < low,
            df['toxicity_prob'].between(low, med, inclusive='left'), 
            df['toxicity_prob'] >= med],
            ['LOW', 'MEDIUM', 'HIGH'],
            default='UKN'
        )

        return df.drop(columns=['toxicity_prob'])
    
    def preprocess_sequences(self, sequences) -> np.ndarray:
        """
        - embedd sequences via ESM-2 model
        - batch size is 4 by default ! (len(sequences) has to be at least 4 !)
        """
        print('generating sequence embeddings')
        return self.embedder.get_embeddings(sequences)

