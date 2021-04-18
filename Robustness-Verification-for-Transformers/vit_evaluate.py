import time

import torch
import torch.nn.functional as F
from torch import optim

from mnist import mnist_test_dataloader
from vit import ViT

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_BATCH_SIZE = 10000
test_loader = mnist_test_dataloader(TEST_BATCH_SIZE)


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=1, heads=4, mlp_dim=128, layer_norm_type="no_var").to(device)

model.load_state_dict(torch.load("mnist_transformer.pt"))
model.eval()
evaluate(model, test_loader, [])




