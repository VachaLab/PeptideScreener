from src.feature_generators.feature_generator import FeatureGenerator

from transformers import EsmModel, AutoTokenizer
from tqdm import tqdm
import torch
import numpy as np

class EmbedderESM2(FeatureGenerator):

    def __init__(self, device = 'cpu', modelname='esm2_t6_8M_UR50D'):
        super().__init__(device)

        self.tokenizer = AutoTokenizer.from_pretrained(f'facebook/{modelname}')
        model = EsmModel.from_pretrained(f'facebook/{modelname}',dtype=torch.float16)
        model = model.to(self.device)
        self.model = model.half()

        self.name = 'ESM2'

    @torch.no_grad
    def get_embeddings(self, seqs, bs = 0, maxlen = 50):

        if bs == 0:
            bs = self.determine_bs(len(seqs))

        embeddings = []

        for i in tqdm(range(0, len(seqs), bs)):

            batch = seqs[i:i+bs]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=maxlen,
                truncation=True,
                padding=True
            ).to(self.device)

            outputs = self.model(**inputs)
            batch_embedds = outputs.last_hidden_state.cpu().numpy()
            batch_embedds = batch_embedds[:, 1:-1, :] # removing <cls> and <eos> tokens
            batch_embedds_mean = batch_embedds.mean(axis=1)

            embeddings.append(batch_embedds_mean)

        embeddings = np.concatenate(embeddings,axis=0)

        return embeddings
    
    def get_features(self, seqs, labels):
        return super().get_features(seqs, labels)