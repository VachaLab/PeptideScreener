from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List

class FeatureGenerator(ABC):

    def __init__(self, device:str='cpu'):

        self.device = device
        

    def get_embeddings(self, seqs:List[str], bs:int = 0, maxlen:int=50) -> np.ndarray:

        """
        Used by PLM-based feature generators
        RETURN: array with embeddings
        """
    
    def get_features(self, seqs:List[str], labels:List) -> pd.DataFrame:

        """
        Used by PCHEM-based feature generators
        RETURN: dataframe with features and labels
        """
    
    def determine_bs(self, seq_count:int) -> int:

        """
        Used by PLM-based feature generators
        """

        bs = 4

        if seq_count >= 1000:
            bs=64
        elif seq_count >= 10000:
            bs=256
        elif seq_count >= 100000:
            bs=512
        
        return bs