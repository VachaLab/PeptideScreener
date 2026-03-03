
"""
TEMPLATE for new screener implementation
"""

from src.screeners.screener import Screener
from pathlib import Path
import pandas as pd

class CustomScreener(Screener):

    def __init__(self, model_path:Path, device:str = 'cpu', seq_header:str = 'sequence'):
        super().__init__(device, seq_header)

        """
        INITIALIZE properties of the object, for example:
        self.model = joblib.load(model_path)
        """
    
    def run_screening(self, df:pd.DataFrame) -> pd.DataFrame:
        
        # get list of sequences from input
        sequences = df[self.header].to_list()

        # run your tool on the sequences

        # add new column(s) with outputs to the dataframe
        # df['your_predictions'] = predictions (predictions is either list or numpy array)

        # return final dataframe
        return df
    

    ### implement any other custom helper functions you want ###