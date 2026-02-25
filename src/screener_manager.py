
from src.screeners.screener import Screener
from src.screeners.toxicity.f_cytotox import CytotoxicityFilter

from src.config import toxicity_clf_path, DEVICE_OPTIONS
from src.utils import get_best_device

import pandas as pd

class ScreenerManager():

    def __init__(self, screeners_dict:dict):

        scr_keys = [key for key in list(screeners_dict.keys()) if screeners_dict[key] == True]
        
        self.screener_list = [self._get_screener_(scrk) for scrk in scr_keys]

    def _get_screener_(self, scr_key:str) -> Screener:

        """
        return corresponding screener based on the src_key
        """

        if scr_key == 'toxicity':
            device = get_best_device(DEVICE_OPTIONS)
            screener = CytotoxicityFilter(model_path=toxicity_clf_path, device=device)
        
        return screener


    def run_complete_screening(self, input_csv:pd.DataFrame) -> pd.DataFrame:
        
        df = input_csv.copy()

        for screener in self.screener_list:
            df = screener.run_screening(df)

        return df