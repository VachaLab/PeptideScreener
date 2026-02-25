from abc import ABC, abstractmethod
import pandas as pd

class Screener(ABC):

    @abstractmethod
    def run_screening(df:pd.DataFrame) -> pd.DataFrame:

        """
        - Run screening on df['sequence'] column
        RETURN: same dataframe with added collumn with predictions
        """