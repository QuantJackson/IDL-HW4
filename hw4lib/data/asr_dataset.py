from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training:
     * Time masking: Masks random time steps
     * Frequency masking: Masks random frequency bands

3. Transcript Processing:
   - Similar to LMDataset transcript handling
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H4Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
   - Provides lengths for packed sequence processing
   - Ensures proper device placement and tensor types

Key Requirements:
- Must maintain feature-transcript alignment
- Must handle variable-length sequences
- Must track maximum lengths for both features and text
- Must implement proper padding for batching
- Must apply SpecAugment only during training
- Must support different normalization strategies
'''

class ASRDataset(Dataset):
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H4Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing.
        Args:
            partition (str): Dataset partition ('train-clean-100', 'dev-clean', or 'test-clean')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
            isTrainPartition (bool): Whether this is the training partition
                                     Used to determine if SpecAugment should be applied.
            global_stats (tuple, optional): (mean, std) computed from training set.
                                          If None and using global_mvn, will compute during loading.
                                          Should only be None for training set.
                                          Should be provided for dev and test sets.
        """
        # TODO: Implement __init__

    
        # Store basic configuration
        self.config    = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # TODO: Get tokenizer ids for special tokens (eos, sos, pad)
        # Hint: See the class members of the H4Tokenizer class
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Set up data paths 
        # TODO: Use root and partition to get the feature directory
        self.fbank_dir   = os.path.join(config['root'], partition, 'fbank')
        
        # TODO: Get all feature files in the feature directory in sorted order  
        self.fbank_files = sorted(os.listdir(self.fbank_dir))
        
        # TODO: Take subset
        subset_size      = self.config.get('subset_size', None)
        self.fbank_files = sorted(os.listdir(self.fbank_dir))[:subset_size] if subset_size else sorted(os.listdir(self.fbank_dir))
        
        # TODO: Get the number of samples in the dataset  
        self.length      = len(self.fbank_files)

        # Case on partition.
        # Why will test-clean need to be handled differently?
        if self.partition != "test-clean":
            # TODO: Use root and partition to get the text directory
            self.text_dir   = os.path.join(self.config["root"], partition, 'text')  

            # TODO: Get all text files in the text directory in sorted order  
            self.text_files = [f for f in sorted(os.listdir(self.text_dir)) if f.endswith('.npy')][:subset_size] if subset_size else [f for f in sorted(os.listdir(self.text_dir)) if f.endswith('.npy')]
        
            
            # Verify data alignment
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # Initialize lists to store features and transcripts
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []
        
        # Initialize counters for character and token counts
        # DO NOT MODIFY
        self.total_chars  = 0
        self.total_tokens = 0
        
        # Initialize max length variables
        # DO NOT MODIFY
        self.feat_max_len = 0
        self.text_max_len = 0
        
        # Initialize Welford's algorithm accumulators if needed for global_mvn
        # DO NOT MODIFY
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            self._count = 0
            self._mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            self._M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)
        else:
            self._count = None
            self._mean = None
            self._M2 = None
        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # TODO: Load features
            # Features are of shape (num_feats, time)
            fbank_filename = self.fbank_files[i]
            feat_path = os.path.join(self.fbank_dir, fbank_filename)
            feat_np = np.load(feat_path)  # shape => (F, T)

            # truncate to config['num_feats']
            F_needed = self.config['num_feats']
            feat_np = feat_np[:F_needed, :]  
            self.feats.append(feat_np)
            self.feat_max_len = max(self.feat_max_len, feat_np.shape[1])

            # Update global statistics if needed (DO NOT MODIFY)
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.from_numpy(feat_np).to(torch.float64)
                time_len = feat_tensor.shape[1]  # number of time steps
                self._count += time_len
                batch_count = feat_tensor.shape[1]     # number of time steps
                count += batch_count
                
                # Update mean and M2 for all time steps at once
                delta = feat_tensor - self._mean.unsqueeze(1)  # (num_feats, time)
                self._mean += delta.mean(dim=1)                # (num_feats,)
                delta2 = feat_tensor - self._mean.unsqueeze(1) # (num_feats, time)
                self._M2 += (delta * delta2).sum(dim=1)        # (num_feats,)

            # NOTE: The following steps are almost the same as the steps in the LMDataset   
            
            if self.partition != "test-clean":
                text_filename = self.text_files[i]
                text_path = os.path.join(self.text_dir, text_filename)
                text_arr = np.load(text_path, allow_pickle=True)
                transcript_str = "".join(text_arr.tolist())

                self.total_chars += len(transcript_str)
                tokenized = self.tokenizer.encode(transcript_str)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                shifted = [self.sos_token] + tokenized
                golden  = tokenized + [self.eos_token]
                self.transcripts_shifted.append(shifted)
                self.transcripts_golden.append(golden)

        # Calculate average characters per token
        # DO NOT MODIFY 
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        
        if self.partition != "test-clean":
            # Verify data alignment
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # Compute final global statistics if needed
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                # Compute variance and standard deviation
                variance = self._M2/(count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = self._mean.float()

        # Initialize SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        DO NOT MODIFY
        """
        # TODO: Implement __len__
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (features, shifted_transcript, golden_transcript) where:
                - features: FloatTensor of shape (num_feats, time)
                - shifted_transcript: LongTensor (time) or None
                - golden_transcript: LongTensor  (time) or None
        """
        # TODO: Load features
        feat_np = self.feats[idx]
        feat = torch.from_numpy(feat_np).float()

        if self.config['norm'] == 'global_mvn':
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            mu = feat.mean(dim=1, keepdim=True)
            sigma = feat.std(dim=1, keepdim=True) + 1e-8
            feat = (feat - mu) / sigma
        else:
            pass

        if self.partition == "test-clean":
            return feat, None, None
        else:
            shifted = self.transcripts_shifted[idx]
            golden  = self.transcripts_golden[idx]
            return feat, torch.LongTensor(shifted), torch.LongTensor(golden)

    def collate_fn(self, batch):
        """
        Return (padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths).
        padded_feats => (B, T, F), etc.
        """
        feats_list    = []
        feat_lengths  = []
        shifted_list  = []
        golden_list   = []
        transcript_lens = []

        for (feat, shifted, golden) in batch:
            # feat => shape (F, T)
            T = feat.shape[1]
            feat_lengths.append(T)
            feats_list.append(feat.transpose(0,1))  # => shape (T, F)

            if self.partition != "test-clean":
                # store transcripts
                shifted_list.append(shifted)
                golden_list.append(golden)
                transcript_lens.append(len(shifted))

        # pad feats => shape (B, max_time, F)
        padded_feats = pad_sequence(feats_list,
                                    batch_first=True,
                                    padding_value=0.0)
        feat_lengths = torch.LongTensor(feat_lengths)

        # transcript handling
        padded_shifted = None
        padded_golden  = None
        transcript_lengths = None

        if self.partition != "test-clean":
            transcript_lengths = torch.LongTensor(transcript_lens)
            padded_shifted = pad_sequence(shifted_list,
                                          batch_first=True,
                                          padding_value=self.pad_token)
            padded_golden  = pad_sequence(golden_list,
                                          batch_first=True,
                                          padding_value=self.pad_token)

        # specaugment if training
        if self.config["specaug"] and self.isTrainPartition:
            # permute to (B, F, T)
            feats_bft = padded_feats.transpose(1,2)  # (B,F,T)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    feats_bft = self.freq_mask(feats_bft)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    feats_bft = self.time_mask(feats_bft)
            # permute back => (B, T, F)
            padded_feats = feats_bft.transpose(1,2)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths

