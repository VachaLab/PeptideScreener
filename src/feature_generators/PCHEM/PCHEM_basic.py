from src.feature_generators.feature_generator import FeatureGenerator

from typing import List
import pandas as pd

class PCHEMBaseline(FeatureGenerator):

    def __init__(self, device = 'cpu'):
        super().__init__(device)
        self.name = 'PCHEM'

        self.charge_dict = {
            'R' : 1,
            'H' : 1,
            'K' : 1,
            'D' : -1,
            'E' : -1
        }

        self.aa_dict = {
            'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 
            'G': 0, 'H': 0, 'I': 0, 'K': 0, 'L': 0, 
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }

    def get_features(self, seqs, labels):

        features_df = self.handcraft_features(seqs, labels)
        return features_df

    ### helper functions ###

    def _get_charge_(self, seq:str):

        charge = 0
        charged_aas = list(self.charge_dict.keys())

        for aa in seq:
            if aa in charged_aas:
                charge += self.charge_dict[aa]
        
        return charge

    def _get_aacounts_(self, seq:str):

        aa_counts = self.aa_dict.copy()

        for aa in seq:
            aa_counts[aa] += 1
        
        sorted_dict =  dict(sorted(aa_counts.items()))
        return pd.DataFrame([sorted_dict])

    def handcraft_features(self, seqs:List[str], labels:List[str], include_charge:bool = True):

        numeric_columns = ['charge', 'len', 'M1']
        numeric_columns.extend(list(self.aa_dict.keys()))
        columns = ['sequence']
        columns.extend(numeric_columns.copy())
        columns.extend(['label'])
        
        # initilize with pre-allocation to optimize the run
        final_df = pd.DataFrame(index=range(len(seqs)),columns=columns)

        for i, seq in enumerate(seqs):
            if include_charge:
                charge = self._get_charge_(seq)
            else:
                charge = 0
            lenght = len(seq)
            start_aa = seq[0]
            start_M:bool = (start_aa == 'M') # methionine on position one was one of 'key' features for OG set

            final_df.iloc[i] = [seq, charge, lenght, start_M] + list(self._get_aacounts_(seq).iloc[0]) + [labels[i]]
        
        #final_df['start_aa'] = final_df['start_aa'].astype('category')
        final_df['label'] = final_df['label'].astype('category')
        final_df[numeric_columns] = final_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        return final_df

