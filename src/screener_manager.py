
from src.screeners.screener import Screener
from src.screeners.toxicity.cytotoxicity_screener import CytotoxicityScreener
from src.screeners.solubility.screener_solubility_will import SolubilityScreenerWill
from src.screeners.solubility.screener_solubility_jana import SolubilityScreenerJana

from src.config import toxicity_clf_path, solubility_will_clf_path, solubility_jana_clf_path, DEVICE_OPTIONS
from src.utils import get_best_device

import pandas as pd
import re

class ScreenerManager():

    def __init__(self, screeners_dict:dict, seq_header:str='sequence'):

        scr_keys = [key for key in list(screeners_dict.keys()) if screeners_dict[key] == True]
        self.header = seq_header
        self.screener_list = [self._get_screener_(scrk) for scrk in scr_keys]

    def _get_screener_(self, scr_key:str) -> Screener:

        """
        return corresponding screener based on the src_key
        """

        if scr_key == 'toxicity':
            device = get_best_device(DEVICE_OPTIONS)
            screener = CytotoxicityScreener(model_path=toxicity_clf_path, device=device, seq_header=self.header)
        if scr_key == 'solubility_will':
            screener = SolubilityScreenerWill(model_path=solubility_will_clf_path, seq_header=self.header)
        if scr_key == 'solubility_jana':
            device = get_best_device(DEVICE_OPTIONS)
            screener = SolubilityScreenerJana(model_path=solubility_jana_clf_path, device=device, seq_header=self.header)
        
        ### TEMPLATE FOR ADDITION OF ANOTHER SCREENER ###
        # if scr_key == 'custom_screener_name':
        #     device = get_best_device(DEVICE_OPTIONS) [OPTIONAL]
        #     screener = CustomScreener(model_path=path_to_model, device=device [OPTIONAL], seq_header=self.header)
        
        return screener

    def curate_sequences(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Curate the DataFrame and return two DataFrames:
        
        1. curated_df: cleaned, uppercase, repeat-expanded sequences using only standard 20 AA
        2. skipped_df: rows that were removed, with 'skip_reason' column explaining why
        
        Cleaning steps:
        - Remove N-terminal Ac-/acetyl notation
        - Remove C-terminal -NH2 notation
        - Remove internal dashes
        - Expand repeat notation e.g. (A5) → AAAAA
        - Convert to uppercase
        - Keep only sequences composed of A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: (curated_df, skipped_df)
        """
        if self.header not in df.columns:
            raise ValueError(f"Column '{self.header}' not found in DataFrame")

        # Work on a copy
        df = df.copy()

        # Add skip_reason column (will be filled only for skipped rows)
        df['skip_reason'] = pd.NA

        # ── Step 1: Basic validity check ────────────────────────────────────────
        df[self.header] = df[self.header].astype(str).str.strip()

        initial_invalid_mask = (
            df[self.header].isna() |
            (df[self.header] == '')
        )
        df.loc[initial_invalid_mask, 'skip_reason'] = 'empty_or_missing'

        # Split early
        candidates = df[~initial_invalid_mask].copy()
        skipped = df[initial_invalid_mask].copy()

        # ── Step 2: Cleaning function ───────────────────────────────────────────
        CLEAN_PATTERNS = {
            "acetyl": re.compile(r"^\s*(AC[- ]|ACE[- ]|ACETYL[- ])", flags=re.I),
            "amid":   re.compile(r"[- ]*(NH2|NHE|AMID)\s*$", flags=re.I),
            "repeat": re.compile(r"\(([ACDEFGHIKLMNPQRSTVWY])\s*(\d+)\)", flags=re.I),
        }

        def clean_and_expand(seq: str) -> tuple[str, str | None]:
            """Return (cleaned_sequence, reason_if_failed)"""
            if not isinstance(seq, str) or not seq:
                return "", "cleaning_failed_empty"
            
            seq = seq.strip().upper()

            orig = seq

            # remove caps (keep the longest string)
            seq = sorted(seq.split("-"), key=len)[-1]

            # # Remove acetyl
            # seq = CLEAN_PATTERNS["acetyl"].sub("", seq)

            # # Remove amid
            # seq = CLEAN_PATTERNS["amid"].sub("", seq)

            # Remove internal dashes
            seq = seq.replace("-", "")

            # Expand repeats
            while (match := CLEAN_PATTERNS["repeat"].search(seq)):
                aa = match.group(1).upper()
                count = int(match.group(2))
                if count < 1 or count > 1000:  # safety limit
                    return orig, "invalid_repeat_count"
                seq = seq[:match.start()] + (aa * count) + seq[match.end():]

            if not seq:
                return "", "cleaning_resulted_in_empty"

            return seq, None

        # Apply cleaning
        clean_results = candidates[self.header].apply(clean_and_expand)
        candidates['cleaned_seq'] = clean_results.str[0]
        candidates['clean_fail_reason'] = clean_results.str[1]

        # Move failed cleaning to skipped
        failed_clean_mask = candidates['clean_fail_reason'].notna()
        failed = candidates[failed_clean_mask].copy()
        failed['skip_reason'] = failed['clean_fail_reason']
        failed = failed.drop(columns=['cleaned_seq', 'clean_fail_reason'], errors='ignore')

        candidates = candidates[~failed_clean_mask].copy()
        candidates[self.header] = candidates['cleaned_seq']
        candidates = candidates.drop(columns=['cleaned_seq', 'clean_fail_reason'], errors='ignore')

        # ── Step 3: Final validation (only standard 20 AA) ──────────────────────
        standard_aa_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

        valid_mask = candidates[self.header].apply(
            lambda x: bool(standard_aa_pattern.fullmatch(x)) if isinstance(x, str) else False
        )

        curated = candidates[valid_mask].copy()
        invalid_aa = candidates[~valid_mask].copy()
        invalid_aa['skip_reason'] = 'non_standard_amino_acids'

        # ── Combine all skipped ─────────────────────────────────────────────────
        skipped = pd.concat([skipped, failed, invalid_aa], ignore_index=True)

        # Clean up columns in skipped
        if 'clean_fail_reason' in skipped.columns:
            skipped = skipped.drop(columns=['clean_fail_reason'], errors='ignore')

        # Final touches
        curated = curated.drop(columns=['skip_reason'], errors='ignore')
        skipped = skipped.reset_index(drop=True)

        return curated, skipped

    def run_complete_screening(self, input_df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        df = input_df.copy()

        print('\n CURATING SEQUENCES \n')
        df, skipped_df = self.curate_sequences(df)

        if df.empty:
            return df, skipped_df

        print('\n STARTING SCREENING \n')
        for screener in self.screener_list:
            df = screener.run_screening(df)

        return df, skipped_df