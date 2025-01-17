import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.optim as optim
from transformers import BartTokenizer, BartForConditionalGeneration
#from Transformer import Transformer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")



input_vocab_list = ['<PAD>','twice', 'and', 'look',  'thrice', 'left', 'turn', 'jump', 'run', 'opposite', 'walk', 'after', 'around', 'right', '<EOS>']
output_vocab_list = ['<PAD>', 'I_WALK', 'I_TURN_LEFT', 'I_RUN', 'I_LOOK', 'I_JUMP', 'I_TURN_RIGHT','<EOS>', "<SOS>"]

input_vocab = {token: idx for idx, token in enumerate(input_vocab_list)}
output_vocab = {token: idx for idx, token in enumerate(output_vocab_list)}

#combined_vocab = list(set(input_vocab_list + output_vocab_list))  # Remove duplicates
#tokenizer.add_tokens(combined_vocab)
# Define special tokens if needed
#tokenizer.pad_token = '<PAD>'
#tokenizer.eos_token = '<EOS>'
#tokenizer.bos_token = '<SOS>'
# Load model and resize embeddings
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
#model.resize_token_embeddings(len(tokenizer))
def tokenizer_bart(input_vocab_list):
    
    tokenizer.add_tokens(input_vocab_list)
    #tokenizer.vocab_size = len(input_vocab_list)
    # Define special tokens if needed
    tokenizer.pad_token = '<PAD>'
    tokenizer.eos_token = '<EOS>'
    tokenizer.bos_token = '<SOS>'
    return tokenizer
    
class SCANData(Dataset):
    def __init__(self, csv_file, input_vocab, output_vocab, tokenizer):
        self.tokenizer = tokenizer
        
        # Handle different input formats (list or DataFrame)
        if isinstance(csv_file, list):
            self.data = pd.DataFrame(csv_file, columns=["IN", "OUT"])
        elif isinstance(csv_file, pd.DataFrame):
            self.data = csv_file
        else:
            raise ValueError("csv_file must be either a list or a pandas DataFrame")

        self.data.reset_index(drop=True, inplace=True)
        self.input_texts = self.data['IN']
        self.target_texts = self.data['OUT']
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize inputs and outputs dynamically
        input_encoding = self.tokenizer(
            input_text, truncation=True, padding=True, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text, truncation=True, padding=True, return_tensors="pt"
        )
        
        # Extract tokenized fields
        input_ids = input_encoding['input_ids'].squeeze(0)  # Remove batch dimension
        input_attention_mask = input_encoding['attention_mask'].squeeze(0)
        
        target_ids = target_encoding['input_ids'].squeeze(0)
        target_attention_mask = target_encoding['attention_mask'].squeeze(0)
        
        # Add the BOS token (<SOS>) at the start of the target sequence
        target_ids = torch.cat([torch.tensor([self.tokenizer.bos_token_id]), target_ids], dim=0)
        target_attention_mask = torch.cat([torch.tensor([1]), target_attention_mask], dim=0)
        
        return {
            'input_ids': input_ids,
            'input_attention_mask': input_attention_mask,
            'target_ids': target_ids,
            'target_attention_mask': target_attention_mask
        }


def collate_fn(batch):
    """
    Collates a batch of data by padding sequences to the maximum length in the batch.
    
    Args:
        batch (list of dicts): The batch containing tokenized input and target data.
        
    Returns:
        dict: Contains padded 'input_ids', 'input_attention_mask', 'target_ids', 'target_attention_mask'.
    """
    input_seqs = [torch.tensor(item['input_ids']) for item in batch]
    input_masks = [torch.tensor(item['input_attention_mask']) for item in batch]
    target_seqs = [torch.tensor(item['target_ids']) for item in batch]
    target_masks = [torch.tensor(item['target_attention_mask']) for item in batch]
    
    # Pad the sequences to the maximum length in the batch
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)  # Pad with tokenizer's PAD token
    input_mask_padded = pad_sequence(input_masks, batch_first=True, padding_value=0)  # Padding mask is 0
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)  # Pad with tokenizer's PAD token
    target_mask_padded = pad_sequence(target_masks, batch_first=True, padding_value=0)  # Padding mask is 0
    
    return {
        'input_ids': input_padded,
        'input_attention_mask': input_mask_padded,
        'target_ids': target_padded,
        'target_attention_mask': target_mask_padded
    }