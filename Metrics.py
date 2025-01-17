import numpy as np

def compute_accuracies(predictions, targets, eos_token_id=50269
):
    """
    Computes token-level and sequence-level accuracy, considering tokens up to the <EOS> token.

    Args:
        predictions (list of list of int): The predicted token IDs for each sequence.
        targets (list of list of int): The target token IDs for each sequence.
        eos_token_id (int): The token ID representing <EOS>. Defaults to 7.

    Returns:
        tuple: A tuple containing:
            - token_accuracy (float): The token-level accuracy.
            - sequence_accuracy (float): The sequence-level accuracy.
    """
    predictions = predictions.cpu().numpy().tolist()
    targets = targets.cpu().numpy().tolist()
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # Truncate predictions and targets up to the first <EOS> token
        pred_truncated = pred[:pred.index(eos_token_id) + 1] if eos_token_id in pred else pred
        target_truncated = target[:target.index(eos_token_id) + 1] if eos_token_id in target else target

        # Compare token-by-token
        seq_correct = True
        for p, t in zip(pred_truncated, target_truncated):
            total_tokens += 1
            if p == t:
                correct_tokens += 1
            else:
                seq_correct = False

        # If lengths differ, the sequence is not fully correct
        if len(pred_truncated) != len(target_truncated):
            seq_correct = False

        # Update sequence-level accuracy
        if seq_correct:
            correct_sequences += 1
            
    # Calculate accuracies
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    sequence_accuracy = correct_sequences / len(targets) if targets else 0.0

    return token_accuracy, sequence_accuracy


def compute_accuracies_single(pred,targets):
    targets = targets[0].cpu().numpy()
    pred = np.array(pred)
    acc = np.sum(np.equal(pred, targets)) / len(targets)
    if acc == 1:
        seq_acc = 1
    else:
        seq_acc = 0
    return acc, seq_acc