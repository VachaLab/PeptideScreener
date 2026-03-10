from src.feature_generators.feature_generator import FeatureGenerator

from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import torch
import numpy as np

class EmbedderBERT(FeatureGenerator):

    def __init__(self, device = 'cpu', modelname='Rostlab/prot_bert'):
        super().__init__(device)

        self.tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=False)
        model = BertModel.from_pretrained(modelname, dtype=torch.float16)
        model = model.to(self.device)
        self.model = model.half()

        self.name = 'PBERT'

    @torch.no_grad
    def get_embeddings(self, seqs, bs = 0, maxlen = 50):

        if bs == 0:
            bs = self.determine_bs(len(seqs))

        embeddings = []
        processed_seqs = [' '.join(seq.strip()) for seq in seqs]

        for i in tqdm(range(0, len(processed_seqs), bs)):

            batch = processed_seqs[i:i+bs]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=maxlen,
                truncation=True,
                padding=True
            ).to(self.device)

            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state

            mask = inputs['attention_mask'].unsqueeze(-1).expand_as(hidden_states).float()
            sum_hidden = torch.sum(hidden_states * mask, dim=1)          # (batch_size, 50)
            num_real_tokens = torch.sum(mask, dim=1).clamp(min=1.0)       # avoid div-by-zero
            mean_pooled = sum_hidden / num_real_tokens   

            embeddings.append(mean_pooled.cpu().numpy())

        embeddings = np.concatenate(embeddings,axis=0)

        return embeddings

    def get_features(self, seqs, labels):
        return super().get_features(seqs, labels)