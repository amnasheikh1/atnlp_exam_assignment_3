import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from Dataset import SCANDataset, collate_fn, tokenizer_bart
from Exp2_Dataset import SCANData
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from torch.nn.functional import pad
from Metrics import compute_accuracies_single
from collections import defaultdict
#from oracle_code import Oracle
tokenizer = tokenizer_bart



def enforce_reproducibility(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

def load_vocab():
    """Load and return input/output vocabulary dictionaries."""
    input_vocab_list = ['<PAD>', 'twice', 'and', 'look', 'thrice', 'left', 'turn', 'jump', 'run',
                        'opposite', 'walk', 'after', 'around', 'right', '<EOS>']
    output_vocab_list = ['<PAD>', 'I_WALK', 'I_TURN_LEFT', 'I_RUN', 'I_LOOK', 'I_JUMP', 'I_TURN_RIGHT',
                         '<EOS>', '<SOS>']

    input_vocab = {token: idx for idx, token in enumerate(input_vocab_list)}
    output_vocab = {token: idx for idx, token in enumerate(output_vocab_list)}

    return input_vocab, output_vocab

def parse_dataset(file_path, input_vocab, output_vocab):
    """Parse the dataset from a given text file."""
    inputs, outputs = [], []
    with open(file_path, 'r') as file:
        for line in file:
            in_part, out_part = line.split('OUT:')
            in_part = in_part.replace('IN:', '').strip()
            out_part = ' '.join(out_part.strip().split()[1:])  # Optional processing
            inputs.append(in_part)
            outputs.append(out_part)
    return pd.DataFrame({'IN': inputs, 'OUT': outputs})

def split_dataset(df, train_ratio=0.8):
    """Split a DataFrame into training and validation subsets."""
    train_size = int(train_ratio * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:].reset_index(drop=True)
    return train_df.reset_index(drop=True), val_df

def create_dataloader(df, input_vocab, output_vocab, batch_size, shuffle, tokenizer):
    """Create a DataLoader for the SCAN dataset."""
    dataset = SCANDataset(df, input_vocab, output_vocab, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def create_dataloader_exp2(df, input_vocab, output_vocab, tokenizer, batch_size, shuffle):
    """Create a DataLoader for the SCAN dataset."""
    # Ensure df is converted to a DataFrame if it's a list
    if isinstance(df, list):
        df = pd.DataFrame(df, columns=["IN", "OUT"])

    dataset = SCANData(df, input_vocab, output_vocab, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def parse_data_exp2(input_file): 
    input_cmds, output_acts = [], []
    with open(input_file, 'r') as file:
        for line in file:
            try:
                input_part, output_part = line.split('OUT:')
                input_part = input_part.replace('IN:', '').strip()
                output_part = output_part.strip()
                input_cmds.append(input_part)
                output_acts.append(output_part)
            except ValueError:
                continue
    return list(zip(input_cmds, output_acts))  # converts input and output into a tuple 

def create_dictionaries_exp2(data):
    # Create dictionaries
    input_length_dict = {}
    output_length_dict = {}
    
    for input_command, output_action in data: 
        input_length = len(input_command.split())
        output_length = len(output_action.split())
        
        if input_length not in input_length_dict:
            input_length_dict[input_length] = []
        input_length_dict[input_length].append((input_command, output_action)) 
        
        if output_length not in output_length_dict:
            output_length_dict[output_length] = []
        output_length_dict[output_length].append((input_command, output_action))
        
    return input_length_dict, output_length_dict

def plot_hist(input_length_dict, output_length_dict):
    # histograms for lengths
    plt.figure(figsize=(12, 6))

    # histogram for input lengths
    plt.subplot(1, 2, 1)
    plt.bar(input_length_dict.keys(), [len(v) for v in input_length_dict.values()], color='skyblue', alpha=0.7)
    plt.xlabel("Input Length ")
    plt.ylabel("Number of Lines")
    plt.title("Histogram of Input Lengths")

    # histogram for output sequence lengths
    plt.subplot(1, 2, 2)
    plt.bar(output_length_dict.keys(), [len(v) for v in output_length_dict.values()], color='purple', alpha=0.7)
    plt.xlabel("Output Sequence Length")
    plt.ylabel("Number of Lines")
    plt.title("Histogram of Output Sequence Lengths")

    plt.tight_layout()
    plt.show()



def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path='metrics.png'):
    """Plot training and validation loss and accuracy."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.plot(epochs, val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='blue')
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

#oracle 
# Oracle function: Evaluates model predictions against ground truth.



# train bart 


def train_bart(model, train_dataloader, optimizer, device, grad_clip=1.0):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        # Move batch to the appropriate device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        input_attention_mask = batch['input_attention_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_attention_mask = batch['target_attention_mask'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids,
                        attention_mask=input_attention_mask,
                        labels=target_ids,
                        decoder_attention_mask=target_attention_mask)
        
        loss = outputs.loss
        

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagate the loss
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()  # Update the model's weights
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def train_bart_exp2(model, input_loaders, output_loaders, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0

    for in_loader, out_loader in zip(input_loaders, output_loaders):
        for in_batch, out_batch in zip(in_loader, out_loader):
            # Move command batch to device
            cmd_src_in = in_batch['input_ids'].to(device)
            cmd_src_attention_mask = in_batch['input_attention_mask'].to(device)
            cmd_tgt = in_batch['target_ids'].to(device)
            cmd_tgt_attention_mask = in_batch['target_attention_mask'].to(device)
            
                        # Forward pass for commands
                        
            in_cmd_outputs = model(
                input_ids= cmd_src_in ,
                attention_mask= cmd_src_attention_mask,
                labels= cmd_tgt,
                decoder_attention_mask= cmd_tgt_attention_mask
            )
            
            in_cmd_loss = in_cmd_outputs.loss

            # Move action batch to device
            act_src_in = out_batch['input_ids'].to(device)
            act_arc_attention_mask = out_batch['input_attention_mask'].to(device)
            act_tgt_in = out_batch['target_ids'].to(device)
            act_tgt_attention_mask = out_batch['target_attention_mask'].to(device)

            # Forward pass for actions
            out_act_outputs = model(
                input_ids= act_src_in,
                attention_mask= act_arc_attention_mask,
                labels= act_tgt_in,
                decoder_attention_mask= act_tgt_attention_mask
            )
            out_act_loss = out_act_outputs.loss

            # averaging the losses
            loss = (in_cmd_loss + out_act_loss) / 2

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate the loss
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()  # Update the model's weights

            total_loss += loss.item()

    avg_train_loss = total_loss / (len(input_loaders) * len(output_loaders))
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss





# validation 

def evaluate_modelbart(model, val_dataloader, optimizer, device, grad_clip=1.0):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = len(val_dataloader.dataset)
    
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
            # Move batch to the appropriate device (GPU or CPU)
            input_ids = batch['input_ids'].to(device)
            input_attention_mask = batch['input_attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_attention_mask = batch['target_attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=input_attention_mask,
                            labels=target_ids,
                            decoder_attention_mask=target_attention_mask)

            loss = outputs.loss
            logits = outputs.logits
            # Calculate token-level accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = target_ids != 0  # Mask for non-padding tokens
            correct_tokens += (predictions == target_ids).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()
            # Calculate sequence-level accuracy
            correct_sequences += (predictions == target_ids).all(dim=1).sum().item()

            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    token_accuracy = correct_tokens / total_tokens * 100 if total_tokens > 0 else 0.0
    sequence_accuracy = correct_sequences / total_sequences * 100 if total_sequences > 0 else 0.0

    print(f"Validation Loss: {avg_loss:.4f}, Token Accuracy: {token_accuracy:.2f}%, Sequence Accuracy: {sequence_accuracy:.2f}%")
    return avg_loss, token_accuracy, sequence_accuracy
 
def oracle_decoder_input(input_ids, target_ids, tokenizer):
  device = 'cuda' if torch.cuda.is_available() else 'cpu' 
  eos_token_id = tokenizer.eos_token_id
  decoder_input_ids = []
  for i in range(target_ids.size(0)):
    row = target_ids[i].tolist()
    eos_index = row.index(eos_token_id) if eos_token_id in row else len(row)
    decoder_input_ids.append(row[:eos_index])
    return torch.tensor(decoder_input_ids).to(device)

# Function to evaluate model on a data loader with oracle
# Oracle assumes ground truth actions are used as context during evaluation
def evaluate_with_oracle(model, cmd_loader, act_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for cmd_batch, act_batch in zip(cmd_loader, act_loader):
            cmd_inputs, cmd_targets = cmd_batch
            act_inputs, act_targets = act_batch

            cmd_inputs = tokenizer(list(cmd_inputs), return_tensors="pt", padding=True, truncation=True).to(device)
            act_targets = tokenizer(list(act_targets), return_tensors="pt", padding=True, truncation=True).to(device)

            oracle_inputs = oracle_decoder_input(cmd_inputs["input_ids"], act_targets["input_ids"], tokenizer)
            outputs = model.generate(cmd_inputs["input_ids"], max_length=50, decoder_input_ids=oracle_inputs)

            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            target_texts = tokenizer.batch_decode(act_targets["input_ids"], skip_special_tokens=True)

            for pred, tgt in zip(predictions, target_texts):
                if pred.strip() == tgt.strip():
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0

# Training loop with oracle
def train_and_evaluate_with_oracle(model, cmd_train_loaders, act_train_loaders, cmd_val_loaders, act_val_loaders, optimizer, epochs):
    results = {
        "train": [],
        "val": []
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()

        for cmd_train_loader, act_train_loader in zip(cmd_train_loaders, act_train_loaders):
            for cmd_batch, act_batch in zip(cmd_train_loader, act_train_loader):
                optimizer.zero_grad()

                cmd_inputs, cmd_targets = cmd_batch
                act_inputs, act_targets = act_batch

                cmd_inputs = tokenizer(list(cmd_inputs), return_tensors="pt", padding=True, truncation=True).to(device)
                act_targets = tokenizer(list(act_targets), return_tensors="pt", padding=True, truncation=True).to(device)

                oracle_inputs = oracle_decoder_input(cmd_inputs["input_ids"], act_targets["input_ids"], tokenizer)

                outputs = model(input_ids=cmd_inputs["input_ids"], decoder_input_ids=oracle_inputs, labels=act_targets["input_ids"])
                loss = outputs.loss

                loss.backward()
                optimizer.step()

        # Evaluate after each epoch for each length group
        train_accuracies = []
        val_accuracies = []

        for cmd_train_loader, act_train_loader, cmd_val_loader, act_val_loader in zip(cmd_train_loaders, act_train_loaders, cmd_val_loaders, act_val_loaders):
            train_accuracy = evaluate_with_oracle(model, cmd_train_loader, act_train_loader)
            val_accuracy = evaluate_with_oracle(model, cmd_val_loader, act_val_loader)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

        results["train"].append(train_accuracies)
        results["val"].append(val_accuracies)

    return results



def plot_results(results, train_lengths, val_lengths):
    plt.figure(figsize=(10, 8))

    # Plot train accuracies
    plt.subplot(2, 1, 1)
    for epoch, accuracies in enumerate(results["train"]):
        plt.bar(train_lengths, accuracies, alpha=0.6, label=f"Epoch {epoch + 1}")

    plt.title("Model Performance by Input Sequence Length")
    plt.xlabel("Input Sequence Length")
    plt.ylabel("Average Accuracy")
    plt.legend()

    # Plot validation accuracies
    plt.subplot(2, 1, 2)
    for epoch, accuracies in enumerate(results["val"]):
        plt.bar(val_lengths, accuracies, alpha=0.6, label=f"Epoch {epoch + 1}")

    plt.title("Model Performance by Target Sequence Length")
    plt.xlabel("Target Sequence Length")
    plt.ylabel("Average Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
 
def evaluate_bart(model, val_dataloader, device, tokenizer, max_len=128):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
            # Move batch to the appropriate device (GPU or CPU)
            input_ids = batch['input_ids'].to(device)
            input_attention_mask = batch['input_attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_attention_mask = batch['target_attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=input_attention_mask,
                            labels=target_ids,
                            decoder_attention_mask=target_attention_mask)

            loss = outputs.loss
            total_loss += loss.item()

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                max_length=max_len
            )

            # Convert IDs to tokens for comparison
            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            decoded_targets = [tokenizer.decode(t, skip_special_tokens=True) for t in target_ids]

            # Calculate token-level accuracy
            for pred, target in zip(generated_ids, target_ids):
                # Pad sequences to the same length
                max_length = max(pred.size(0), target.size(0))
                pred_padded = pad(pred, (0, max_length - pred.size(0)), value=tokenizer.pad_token_id)
                target_padded = pad(target, (0, max_length - target.size(0)), value=tokenizer.pad_token_id)

                # Count correct tokens
                correct_tokens += torch.sum(pred_padded == target_padded).item()
                total_tokens += target_padded.size(0)

            # Calculate sequence-level accuracy
            for pred_seq, target_seq in zip(decoded_preds, decoded_targets):
                if pred_seq == target_seq:
                    correct_sequences += 1
                total_sequences += 1

    # Calculate metrics
    avg_val_loss = total_loss / len(val_dataloader)
    token_accuracy = correct_tokens / total_tokens
    sequence_accuracy = correct_sequences / total_sequences

    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Token-Level Accuracy: {token_accuracy:.4f}")
    print(f"Sequence-Level Accuracy: {sequence_accuracy:.4f}")

    return avg_val_loss, token_accuracy, sequence_accuracy


def evaluate_bart_seq_lenght(model, val_input_loaders, val_output_loaders, device, tokenizer, start_token_id, eos_token_id):
    
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_seqs = 0 
    total_seqs = 0
    
    # Dictionaries to store correct predictions and total predictions for each length
    src_length_acc = defaultdict(lambda: [0, 0])  # [correct_tokens, total_tokens]
    tgt_length_acc = defaultdict(lambda: [0, 0])  # [correct_tokens, total_tokens]
    
    with torch.no_grad():
        for in_loader, out_loader in zip(val_input_loaders, val_output_loaders):
            for in_batch, out_batch in zip(in_loader, out_loader):
                # Command evaluation
                cmd_src_in = in_batch['input_ids'].to(device)
                cmd_src_attention_mask = in_batch['input_attention_mask'].to(device)
                cmd_tgt_in = in_batch['target_ids'].to(device)

                in_cmd_outputs = model(
                    input_ids=cmd_src_in,
                    attention_mask=cmd_src_attention_mask,
                    labels=cmd_tgt_in
                )
                loss = in_cmd_outputs.loss
                total_loss += loss.item()
                logits = in_cmd_outputs.logits

                predictions = torch.argmax(logits, dim=-1)
                in_cmd_lengths = (cmd_tgt_in != tokenizer.pad_token_id).sum(dim=1)
                
                # Token and sequence level accuracy for commands
                for i in range(cmd_tgt_in.size(0)):
                    seq_correct = True
                    for j in range(in_cmd_lengths[i]):
                        if cmd_tgt_in[i, j] not in [start_token_id, eos_token_id, tokenizer.pad_token_id]:
                            length = in_cmd_lengths[i].item()
                            src_length_acc[length][1] += 1  # Total tokens
                            total_tokens += 1
                            if predictions[i, j] == cmd_tgt_in[i, j]:
                                src_length_acc[length][0] += 1  # Correct tokens
                                total_correct_tokens += 1
                            else:
                                seq_correct = False
                    if seq_correct:
                        total_correct_seqs += 1
                    total_seqs += 1

                
                
                # Action evaluation
                act_src_in = out_batch['input_ids'].to(device)
                act_src_attention_mask = out_batch['input_attention_mask'].to(device)
                act_tgt_in= out_batch['target_ids'].to(device)

                out_act_outputs = model(
                    input_ids=act_src_in,
                    attention_mask=act_src_attention_mask,
                    labels=act_tgt_in
                )
                loss = out_act_outputs.loss
                total_loss += loss.item()
                logits = out_act_outputs.logits

                predictions = torch.argmax(logits, dim=-1)
                out_act_lengths = (act_tgt_in != tokenizer.pad_token_id).sum(dim=1)
                
                # Token and sequence level accuracy for actions
                for i in range(act_tgt_in.size(0)):
                    seq_correct = True
                    for j in range(out_act_lengths[i]):
                        if act_tgt_in[i, j] not in [start_token_id, eos_token_id, tokenizer.pad_token_id]:
                            length = out_act_lengths[i].item()
                            tgt_length_acc[length][1] += 1  # Total tokens
                            total_tokens += 1
                            if predictions[i, j] == act_tgt_in[i, j]:
                                tgt_length_acc[length][0] += 1  # Correct tokens
                                total_correct_tokens += 1
                            else:
                                seq_correct = False
                    if seq_correct:
                        total_correct_seqs += 1
                    total_seqs += 1


                

    # Compute accuracy per length
    src_length_acc = {length: correct / total for length, (correct, total) in src_length_acc.items()}
    tgt_length_acc = {length: correct / total for length, (correct, total) in tgt_length_acc.items()}
    # Compute overall metrics
    avg_loss = total_loss / (len(val_input_loaders) + len(val_output_loaders))
    token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    sequence_accuracy = total_correct_seqs / total_seqs if total_seqs > 0 else 0

    return avg_loss, token_accuracy, sequence_accuracy, src_length_acc, tgt_length_acc

# Training and Validation
def train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=10, grad_clip=1.0, tgt_pad_idx=0
):
    """Train and validate the Transformer model."""
    model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch [{epoch}/{num_epochs}]")

        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        with tqdm(total=len(train_loader), desc="Training") as pbar:
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)

                tgt_input, tgt_output = target_ids[:, :-1], target_ids[:, 1:] # use teacher forcing

                optimizer.zero_grad()
                logits = model(input_ids, tgt_input)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=-1)
                mask = tgt_output != tgt_pad_idx
                train_correct += (preds[mask] == tgt_output[mask]).sum().item() # token level accuracy
                train_total += mask.sum().item()

                pbar.set_postfix(loss=loss.item(), accuracy=train_correct / train_total)
                pbar.update()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_correct / train_total)
        print(f"Training Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}")

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Validation") as pbar:
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    target_ids = batch["target_ids"].to(device)

                    tgt_input, tgt_output = target_ids[:, :-1], target_ids[:, 1:]
                    logits = model(input_ids, tgt_input)

                    loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
                    val_loss += loss.item()

                    preds = logits.argmax(dim=-1)
                    mask = tgt_output != tgt_pad_idx
                    val_correct += (preds[mask] == tgt_output[mask]).sum().item()
                    val_total += mask.sum().item()

                    pbar.set_postfix(loss=loss.item(), accuracy=val_correct / val_total)
                    pbar.update()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / val_total)
        print(f"Validation Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.4f}")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# Evaluation
def evaluate_model(model, data_loader, criterion, device, tgt_pad_idx, eos_idx):
    """Evaluate the model on a test dataset."""
    model.eval()
    test_loss, test_correct, test_total, sequence_correct, total_sequences = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            tgt_input, tgt_output = target_ids[:, :-1], target_ids[:, 1:]
            logits = model(input_ids, tgt_input)

            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            test_loss += loss.item()

            # Get predictions
            preds = logits.argmax(dim=-1)

            # Token level accuracy (ignoring padding tokens)
            mask = tgt_output != tgt_pad_idx
            test_correct += (preds[mask] == tgt_output[mask]).sum().item()
            test_total += mask.sum().item()

            # Sequence-level accuracy (truncate EOS and ignore padding)
            batch_sequence_correct = 0
            for pred, target in zip(preds, tgt_output):
                # Truncate at EOS for prediction
                if eos_idx in pred:
                    pred = pred[:(pred == eos_idx).nonzero(as_tuple=True)[0][0]]

                # Truncate at EOS for target
                if eos_idx in target:
                    target = target[:(target == eos_idx).nonzero(as_tuple=True)[0][0]]

                # Remove padding from target
                target = target[target != tgt_pad_idx]

                # Compare processed sequences
                if torch.equal(pred, target):
                    print("match", pred, target)
                    batch_sequence_correct += 1


            sequence_correct += batch_sequence_correct
            total_sequences += input_ids.size(0)


    token_accuracy = test_correct / test_total
    sequence_accuracy = sequence_correct / total_sequences
    avg_loss = test_loss / len(data_loader)

    return avg_loss, token_accuracy, sequence_accuracy

def greedy_decode(model, src_in, max_len, start_token_id, eos_token_id):
    """Greedy decoding for sequence generation."""
    model.eval()
    with torch.no_grad():
        # Create a tensor to store the decoded sequence
        decoded_sequences = torch.full(
            (src_in.size(0), 1),  # Shape: (batch_size, 1)
            start_token_id,
            dtype=torch.long,
            device=src_in.device
        )

        for _ in range(max_len):
            # Ensure decoded_sequences is contiguous before passing to the model
            decoded_sequences = decoded_sequences.contiguous()  

            # Pass the input sequence to the model
            output = model(src_in, decoder_input_ids=decoded_sequences)  # Updated to decoder_input_ids

            # Get the predicted token (the one with the highest probability)
            predicted_token = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

            # Add the predicted token to the decoded sequence
            decoded_sequences = torch.cat([decoded_sequences, predicted_token], dim=1)

            # Check if all sequences have generated the EOS token
            if (decoded_sequences == eos_token_id).any(dim=1).all():
                break

    return decoded_sequences  # Return the decoded sequence

def prepare_partial_teacher_forcing_input(model, src_in, tgt_in, start_token_id):
    """
    Prepares the decoder input for partial teacher forcing during training.
    
    Args:
        model: The trained model.
        src_in (torch.Tensor): Input token IDs (batch_size, src_len).
        tgt_in (torch.Tensor): Target token IDs (batch_size, tgt_len).
        start_token_id (int): ID for the <START> token.
    
    Returns:
        torch.Tensor: Decoder input IDs with partial teacher forcing applied.
    """
    batch_size, tgt_len = tgt_in.size()
    decoder_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long).to(src_in.device)
    
    for i in range(1, tgt_len):
        # Use teacher forcing for odd-indexed tokens and model predictions for even-indexed tokens
        if i % 2 == 1:
            next_token = tgt_in[:, i].unsqueeze(1)  # Ground-truth token
        else:
            # Use the model's prediction
            output = model(src_in, decoder_input)  # (batch_size, seq_len, vocab_size)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)  # Predicted token
        
        # Append the token to the decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
    
    return decoder_input

def greedy_decode_single(model, src_in, target, start_token_id, eos_token_id):
    """
    Performs greedy decoding for a single input sequence using the provided model.

    Args:
        model: The trained model to use for decoding.
        src_in: The input sequence tensor.
        target: The target sequence tensor.
        start_token_id: The ID of the start token.
        eos_token_id: The ID of the end-of-sequence token.

    Returns:
        The decoded output sequence tensor.
    """
    model.eval()
    with torch.no_grad():
        current_sequence = torch.tensor([[start_token_id]], dtype=torch.long, device=src_in.device)
        
        for _ in range(target.size(1) - 1):  # Iterate up to the length of the target sequence - 1
            # Pass the input sequence and the current sequence to the model.
            # ensure the current_sequence is contiguous
            output = model(input_ids=src_in.contiguous(), decoder_input_ids=current_sequence.contiguous())
            # Get the next token from the model's output.
            next_token_probs = output.logits[0, -1, :] # output is a Seq2SeqLMOutput object
            next_token = torch.argmax(next_token_probs).unsqueeze(0).unsqueeze(0)  
            
            # Append the next token to the current sequence.
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
            
            # If the end-of-sequence token is predicted, stop decoding.
            if next_token.item() == eos_token_id:
                break
            
    return current_sequence

def greedy_decode_single_oracle(model, src_in, target, start_token_id,eos_token_id):
    """
    Performs greedy decoding for a single input sequence with special handling for token `7`.


    """
    decoded_sequence = [start_token_id]  # Start decoding with the <START> token
    max_len = len(target[0])
    for step in range(max_len):
        # Prepare the input for the model
        current_sequence = torch.tensor(decoded_sequence, dtype=torch.long).unsqueeze(0).to(src_in.device)
        
        # Pass the input sequence to the model
        output = model(src_in, current_sequence)  # Shape: (1, seq_len, vocab_size)
        
        # Select the logits for the next token
        logits = output[0, -1, :]  # Shape: (vocab_size,)
        
        # Get the top 2 token predictions
        top2_tokens = logits.topk(2).indices  # Get the top 2 token IDs
        next_token = top2_tokens[0].item()  # Default to the highest logit
        
        # Always set the last to eos 
        if step == max_len - 1:
            next_token = eos_token_id
            
        elif next_token == eos_token_id and step != max_len - 1:
            next_token = top2_tokens[1].item()  # Use the second-highest logit
        
        # Add the selected token to the sequence
        decoded_sequence.append(next_token)

    # Remove the initial <START> token and return the decoded sequence
    return decoded_sequence[1:]
def evaluate_bart_exp_2(model, eval_loader, device, start_token_id=3, eos_token_id=50269):
  model.eval()
  
  with torch.no_grad():
    token_acc = 0
    seq_acc = 0
    input_length_metrics = {}
    target_length_metrics = {}
    
    for batch in tqdm(eval_loader):
      src_in = batch['input_ids'].to(device)
      attention_mask = batch['input_attention_mask'].to(device)
      tgt_in = batch['target_ids'].to(device)
      decoder_attention_mask = batch['target_attention_mask'].to(device)

      # forward pass 
      outputs = model.generate(
        src_in = batch['input_ids'],
        attention_mask=attention_mask, 
        max_length= tgt_in.size(1),    # to match predicted length to target length 
        decoder_start_token=start_token_id,
        eos_token_id=eos_token_id,
        num_beams= 1, # greedy decoding used for consistency 
      )

      token_acc_batch, seq_acc_batch = compute_accuracies_single(outputs, tgt_in[:, 1:])
      token_acc += token_acc_batch 
      seq_acc += seq_acc_batch

      input_lengths = (src_in != 0).sum(dim=1).cpu().numpy()  # Non-zero token counts
      target_lengths = (tgt_in != 0).sum(dim=1).cpu().numpy()

      for length in input_lengths:
        if length not in input_length_metrics:
          input_length_metrics[length] = []
        input_length_metrics[length].append(token_acc_batch)

      for length in target_lengths:
        if length not in target_length_metrics:
          target_length_metrics[length] = []
        target_length_metrics[length].append(token_acc_batch)

  avg_token_acc = token_acc / len(eval_loader)
  avg_seq_acc = seq_acc / len(eval_loader)
  return avg_token_acc, avg_seq_acc
  
      

def evaluate_model_exp2(model, eval_dataloader, device, start_token_id=8, eos_token_id=7):
    #model.load_state_dict(torch.load("Archive/models/best_model_experiment2.pth"))
    model.eval()

    with torch.no_grad():
        token_acc = 0
        seq_acc = 0
        input_length_metrics = {}
        target_length_metrics = {}
        for batch in tqdm(eval_dataloader):
            src_in = batch['input_ids'].to(device).contiguous()
            tgt_in = batch['target_ids'].to(device).contiguous()

            input_lengths = (src_in != 0).sum(dim=1).cpu().numpy() - 1
            target_lengths = (tgt_in != 0).sum(dim=1).cpu().numpy() - 2

            new_output = greedy_decode_single(
                model, 
                src_in, 
                tgt_in[:, :-1], 
                start_token_id=start_token_id, 
                eos_token_id=eos_token_id
            )

            token_acc_batch, seq_acc_batch = compute_accuracies_single(new_output, tgt_in[:, 1:])
            token_acc += token_acc_batch
            seq_acc += seq_acc_batch

            for length in input_lengths:
                if length not in input_length_metrics:
                    input_length_metrics[length] = []
                input_length_metrics[length].append(token_acc_batch)

            for length in target_lengths:
                if length not in target_length_metrics:
                    target_length_metrics[length] = []
                target_length_metrics[length].append(token_acc_batch)
    token_accuracy = token_acc / len(eval_dataloader)
    sequence_accuracy = seq_acc / len(eval_dataloader)

    print(f"Test Token Acc: {token_accuracy} Test Seq Acc: {sequence_accuracy}")
    
    input_length_metrics = {k: np.mean(v) for k, v in input_length_metrics.items()}
    target_length_metrics = {k: np.mean(v) for k, v in target_length_metrics.items()}
    
    plot_metrics(input_length_metrics, 'tokens', 'Model Performance by Input Sequence Length', save_path= "Exp2_InputToken" )
    plot_metrics(target_length_metrics, 'tokens', 'Model Performance by Target Sequence Length', save_path= "Exp2_TargetToken")
    
    return token_accuracy, sequence_accuracy

def evaluate_model_exp2_oracle(model, eval_dataloader, device, start_token_id=8, eos_token_id=7):
        # Load the best model
    model.load_state_dict(torch.load("Archive/models/best_model_experiment2.pth"))
    model.eval()

    # Evaluate the model on the test set
    with torch.no_grad():
        token_acc = 0
        seq_acc = 0
        input_length_metrics_oracle = {}
        target_length_metrics_oracle = {}
        input_length_metrics_oracle_seq = {}
        target_length_metrics_oracle_seq = {}
        for batch in tqdm(eval_dataloader):
            src_in = batch['input_ids'].to(device)
            tgt_in = batch['target_ids'].to(device)


            # Compute lengths
            input_lengths = (src_in != 0).sum(dim=1).cpu().numpy() - 1 # Subtract 1 to exclude the <eos> token
            target_lengths = (tgt_in != 0).sum(dim=1).cpu().numpy() - 2 # Subtract 2 to exclude the <sos> and <eos> tokens

            new_output = greedy_decode_single_oracle(
                model, 
                src_in, 
                tgt_in[:,:-1], # Maximum length of the target sequence
                start_token_id=start_token_id, 
                eos_token_id=eos_token_id
            )

            # Compute accuracy
            token_acc_batch, seq_acc_batch = compute_accuracies_single(new_output, tgt_in[:, 1:])
            token_acc += token_acc_batch
            seq_acc += seq_acc_batch
            # Store accuracies by input and target lengths
            for input_length, target_length in zip(input_lengths,target_lengths):
                
                if input_length not in input_length_metrics_oracle:
                    input_length_metrics_oracle[input_length] = []
                input_length_metrics_oracle[input_length].append(token_acc_batch)
                
                if input_length not in input_length_metrics_oracle_seq:
                    input_length_metrics_oracle_seq[input_length] = []
                input_length_metrics_oracle_seq[input_length].append(seq_acc_batch)
                
                if target_length not in target_length_metrics_oracle:
                    target_length_metrics_oracle[target_length] = []
                target_length_metrics_oracle[target_length].append(token_acc_batch)
                
                if target_length not in target_length_metrics_oracle_seq:
                    target_length_metrics_oracle_seq[target_length] = []
                target_length_metrics_oracle_seq[target_length].append(seq_acc_batch)
    token_accuracy = token_acc / len(eval_dataloader)
    sequence_accuracy = seq_acc / len(eval_dataloader)

    print(f"Test Token Acc: {token_accuracy} Test Seq Acc: {sequence_accuracy}")

    input_length_metrics_oracle = {k: np.mean(v) for k, v in input_length_metrics_oracle.items()}
    target_length_metrics_oracle = {k: np.mean(v) for k, v in target_length_metrics_oracle.items()}
    input_length_metrics_oracle_seq = {k: np.mean(v) for k, v in input_length_metrics_oracle_seq.items()}
    target_length_metrics_oracle_seq = {k: np.mean(v) for k, v in target_length_metrics_oracle_seq.items()}
    
    plot_metrics(input_length_metrics_oracle, 'tokens', 'Model Performance by Input Sequence Length with Oracle', save_path= "Exp2_InputOracleToken" )
    plot_metrics(target_length_metrics_oracle, 'tokens', 'Model Performance by Target Sequence Length with Oracle', save_path= "Exp2_TargetOracleToken")
    plot_metrics(input_length_metrics_oracle_seq, 'sequence', 'Model Performance by Input Sequence Length with Oracle', save_path= "Exp2_InputOracleSeq")
    plot_metrics(target_length_metrics_oracle_seq, 'sequence', 'Model Performance by Target Sequence Length with Oracle', save_path= "Exp2_TargetOracleSeq")
    
    return token_accuracy, sequence_accuracy
    
def plot_metrics(data_dict, unit, title, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.bar(data_dict.keys(), data_dict.values(), color='skyblue', alpha=1)
    plt.title(title)
    plt.xlabel('Sequence Length')
    plt.ylabel(f'Average Accuracy ({unit})')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig("images/"+save_path)