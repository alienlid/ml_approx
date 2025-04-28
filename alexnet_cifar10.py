import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import approx_methods
import argparse

# custom AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 128, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            # 128 x 32 x 32
            nn.MaxPool2d(kernel_size=2),
            # 128 x 16 x 16
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            # 128 x 16 x 16
            nn.MaxPool2d(kernel_size=2),
            # 128 x 8 x 8
            nn.Conv2d(128, 192, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            # 192 x 8 x 8
            nn.Conv2d(192, 192, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            # 192 x 8 x 8
            nn.Conv2d(192, 128, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            # 128 x 8 x 8
            nn.MaxPool2d(kernel_size=2),
            # 128 x 4 x 4
            nn.Flatten(),
            # 2048
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 1024
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 256
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        logits = self.features(x)
        probs = F.softmax(logits, dim=1)
        return logits

def actual_Ab_l2_dists(f):
    l2_dists = []
    model.eval()
    for c in tqdm(cs):
        weight_approx = f(weight, c)
        diff = torch.from_numpy(weight - weight_approx).cuda()
        l2_dist = 0
        for x, _ in test_loader:
            x = x.cuda()
            b = model.features[:21](x)
            error = b @ diff.T
            l2_dist += torch.sum(torch.norm(error, dim=1) / torch.norm(b, dim=1))
        
        l2_dist /= len(test_dataset) * np.linalg.norm(weight)
        l2_dists.append(l2_dist.item())
        tqdm.write(f"method: {args.method}, c: {c}, l2_dist: {l2_dist.item()}")

    return l2_dists


def actual_Ab_accs(f):
    accs = []
    model.eval()
    for c in tqdm(cs):
        weight_approx = f(weight, c)
        model.features[21].weight.data = torch.from_numpy(weight_approx).cuda()
        total = 0
        correct = 0
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            total += len(y)
            correct += torch.sum(pred.argmax(1) == y)
        accs.append((correct / total).item())
        tqdm.write(f"method: {args.method}, c: {c}, accuracy: {accs[-1]}")

    return accs

if __name__ == "__main__":
    # get method from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    args = parser.parse_args()
    func = getattr(approx_methods, args.method)
    print(f"method: {args.method}, metric: {args.metric}")

    # load dataset
    test_dataset = datasets.CIFAR10(
        root='/workspace/ml_approx/CIFAR10',
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        num_workers=2,
        shuffle=False,
    )
    
    # load model
    model = AlexNet(num_classes=10)
    model.load_state_dict(torch.load("/workspace/ml_approx/modelweights"))
    model.cuda()
    weight = model.features[21].weight.data.cpu().numpy()

    # compression ratios
    # cs = np.concat((
    #     np.arange(0, 128, 8),
    #     np.arange(128, 192, 4),
    #     np.arange(192, 224, 2),
    #     np.arange(224, 257, 1),
    # )) / 256
    cs = np.linspace(0, 1, 257)

    if args.metric == "l2":
        l2_dists = actual_Ab_l2_dists(func)
        np.save(f"/workspace/ml_approx/data/alexnet_cifar10/aA_ab_l2_{args.method}.npy", l2_dists)
    elif args.metric == "acc":
        accs = actual_Ab_accs(func)
        np.save(f"/workspace/ml_approx/data/alexnet_cifar10/aA_ab_acc_{args.method}.npy", accs)
    else:
        raise ValueError(f"Unknown metric: {args.metric}")
    
    print("Done!")