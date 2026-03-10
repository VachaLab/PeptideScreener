from src.screeners.screener import Screener
import pandas as pd
import numpy as np
from typing import Dict, List
import re
from pathlib import Path
import joblib
from localcider.sequenceParameters import SequenceParameters
from sparrow import Protein
import xgboost


# Precompile regexes once (already good, just moved outside class)
CLEAN_ACETYL = re.compile(r"^\s*Ac-", flags=re.I)
CLEAN_AMID   = re.compile(r"-NH2\s*$", flags=re.I)
CLEAN_REPEAT = re.compile(r"\(([ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy])(\d+)\)", flags=re.I)


def fast_expand_repeats(seq: str) -> str:
    """Faster repeat expansion using list + join instead of repeated string slicing"""
    if '(' not in seq:
        return seq

    parts = []
    pos = 0

    for m in CLEAN_REPEAT.finditer(seq):
        start, end = m.span()
        aa, count = m.group(1).upper(), int(m.group(2))
        parts.append(seq[pos:start])
        parts.append(aa * count)
        pos = end

    parts.append(seq[pos:])
    return "".join(parts)


class SolubilityScreenerWill(Screener):
    def __init__(self, model_path: Path, seq_header: str = 'sequence'):
        super().__init__(seq_header=seq_header)
        model_dict = joblib.load(model_path)
        self.model = model_dict['model']
        
        # Cache pipeline steps for faster access
        self.scaler = self.model.named_steps['scaler']
        self.xgb = self.model.named_steps['xgb']


    def clean_sequence(self, raw: str) -> str:
        """Optimized cleaning pipeline"""
        seq = CLEAN_ACETYL.sub("", raw)
        seq = CLEAN_AMID.sub("", seq)
        seq = seq.replace("-", "")          # still fast
        seq = fast_expand_repeats(seq)
        return seq.strip().upper()


    def calc_sequence_features(self, seq: str) -> Dict[str, float]:
        """Same features, but we try to minimize object creation overhead"""
        sp = SequenceParameters(seq)
        prot = Protein(seq)

        # You can experiment with caching amino acid fractions if many sequences are similar
        aaf = sp.get_amino_acid_fractions()

        return {
            "fracpol": sum(aaf[x] for x in "QNSTGCH"),
            "dispro": sp.get_fraction_disorder_promoting(),
            "isopoi": sp.get_isoelectric_point(),
            "e2e_scaled": prot.predictor.end_to_end_distance(use_scaled=True),
            "rg_scaled": prot.predictor.radius_of_gyration(use_scaled=True),
            "scaling_exponent": prot.predictor.scaling_exponent(),
            "prefactor": prot.predictor.prefactor(),
        }


    def run_screening(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main optimizations are here:
        - vectorized cleaning (list comp → still needed, but faster inner func)
        - avoid redundant np.array(…) call
        - avoid .copy()
        - direct pipeline access
        - single dtype conversion at the end
        """
        sequences = df[self.seq_header].tolist()

        # Stage 1: clean (parallelize this in production with multiprocessing / joblib.Parallel)
        cleaned_seqs = [self.clean_sequence(seq) for seq in sequences]

        # Stage 2: feature extraction
        features_list = [self.calc_sequence_features(seq) for seq in cleaned_seqs]

        # Stage 3: create DataFrame
        features_df = pd.DataFrame(features_list, index=df.index)

        # Stage 4: model inference
        X_scaled = self.scaler.transform(features_df)
        probabilities = self.xgb.predict_proba(X_scaled)[:, 1]

        print('TEST')
        df['will_solubility_prob'] = probabilities.astype(np.float32)
        # fix formatting issues
        df['will_solubility_prob'] = df['will_solubility_prob'].round(3).map('{:.3f}'.format)

        return df