import torch
from torch import nn
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoader import MovieDataset
from LSTM import LSTMModel
from GloveEmbed import _get_embedding
import os
import time


def _save_checkpoint(ckp_path, model, epoch, global_step, optimizer):
    os.makedirs(os.path.dirname(ckp_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, ckp_path)


def run_experiment(pretrain, embedding_dim, glove_file=None):
    print(f"\n=== Running experiment: {'GloVe' if pretrain else 'Random'} embedding, dim={embedding_dim} ===")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    Batch_size = 64
    n_layers = 2
    input_len = 250
    hidden_dim = 100
    output_size = 1
    num_epoches = 30
    learning_rate = 0.001
    clip = 5

    # Load vocab
    with open('tokens2index.json') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    # Load embedding matrix
    if pretrain and glove_file is not None:
        embedding_matrix = _get_embedding(glove_file, tokens2index, embedding_dim)
    else:
        embedding_matrix = None

    # Build model
    model = LSTMModel(
        vocab_size=vocab_size,
        output_size=output_size,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        input_len=input_len,
        pretrain=pretrain
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


    # Data
    train_loader = DataLoader(MovieDataset('training_processed.csv'), batch_size=Batch_size, shuffle=True)
    test_loader = DataLoader(MovieDataset('test_processed.csv'), batch_size=Batch_size, shuffle=False)

    # Training
    model.train()
    for epoch in range(num_epoches):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_out = model(x)
            loss = criterion(y_out, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss / len(train_loader):.4f}")

    # Testing
    print("Model testing on test data...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_true in test_loader:
            x_batch, y_true = x_batch.to(device), y_true.to(device)

            logits = model(x_batch)  # raw output (no sigmoid)
            probs = torch.sigmoid(logits)  # apply sigmoid for prediction
            y_pred = torch.round(probs)    # convert to 0 or 1

            correct += (y_pred == y_true).sum().item()
            total += y_true.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



def main():
    time_start = time.time()

    # Run all 3 experiments
    acc_random = run_experiment(pretrain=False, embedding_dim=50)
    acc_glove200 = run_experiment(pretrain=True, embedding_dim=200, glove_file='/Users/huyennguyen/Downloads/glove/glove.6B.200d.txt')
    acc_glove300 = run_experiment(pretrain=True, embedding_dim=300, glove_file='/Users/huyennguyen/Downloads/glove/glove.6B.300d.txt')

    time_end = time.time()
    print(f"\nTotal time: {(time_end - time_start) / 60:.2f} mins")


if __name__ == '__main__':
    main()
