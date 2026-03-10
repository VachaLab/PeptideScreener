from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

from src.config import DEVICE_OPTIONS
from src.utils import get_best_device

class Screener(ABC):

    def __init__(self, device:str=None, seq_header:str='sequence', label_header:str='label'):

        self.device = device
        self.seq_header = seq_header

        if not self.device:
            self.device = get_best_device(DEVICE_OPTIONS)
        
        ### FOR TRAINING PURPOSES ###
        self.label_header = label_header
        

    def run_screening(df:pd.DataFrame) -> pd.DataFrame:

        """
        - Run screening on df['sequence'] column
        RETURN: same dataframe with added collumn with predictions
        """

    ### FOR TRAINING PURPOSES ###
    
    def gen_config(self) -> None:
        
        """
        generate config.yaml to later build the screener in PeptideScreener tool
        """
    
    def design_screener(self, train_df:pd.DataFrame, val_df:pd.DataFrame, output_dir:Path) -> None:

        """
        run complete design of new screener
        - train model
        - evaluate model
        - save classifier
        - generate configuration for later usage in PeptideScreener
        """
    
    def train_eval(self) -> None:
        """
        run traininig and evaluation
        """