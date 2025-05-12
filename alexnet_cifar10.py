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

def actual_Ab_l2s_and_cossims(f):
    l2s = []
    cossims = []
    model.eval()
    for c in tqdm(cs):
        weight_approx = f(weight, c)
        # diff = torch.from_numpy(weight - weight_approx).cuda()
        weight_approx = torch.from_numpy(weight_approx).cuda()
        l2 = 0
        cossim = 0
        for x, _ in test_loader:
            x = x.cuda()
            b = model.features[:21](x)
            error = b @ (weight_cuda - weight_approx).T
            l2 += torch.sum(torch.norm(error, dim=1) / torch.norm(b, dim=1))
            cossim += torch.sum(torch.nn.functional.cosine_similarity(
                b @ weight_cuda.T,
                b @ weight_approx.T,
                dim=1,
            ))
        
        l2 /= len(test_dataset) * np.linalg.norm(weight)
        cossim /= len(test_dataset)
        l2s.append(l2.item())
        cossims.append(cossim.item())
        tqdm.write(f"method: {args.method}, c: {c}, l2: {l2.item()}, cossim: {cossim.item()}    ")

    return l2s, cossims

def actual_Ab_accs_and_losses(f):
    accs = []
    losses = []
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    for c in tqdm(cs):
        weight_approx = f(weight, c)
        model.features[21].weight.data = torch.from_numpy(weight_approx).cuda()
        total = 0
        correct = 0
        total_loss = 0.0
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            total += len(y)
            correct += torch.sum(pred.argmax(1) == y)
            total_loss += loss_fn(pred, y).item()
        accs.append((correct / total).item())
        losses.append(total_loss / total)
        tqdm.write(f"method: {args.method}, c: {c}, acc: {accs[-1]}, loss: {losses[-1]}")

    return accs, losses

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
    weight_cuda = torch.from_numpy(weight).cuda() # dumb code lmao

    # compression ratios
    # cs = np.concat((
    #     np.arange(0, 128, 8),
    #     np.arange(128, 192, 4),
    #     np.arange(192, 224, 2),
    #     np.arange(224, 257, 1),
    # )) / 256
    cs = np.linspace(0, 1, 257)

    if args.metric == "l2":
        l2s, cossims = actual_Ab_l2s_and_cossims(func)
        # np.save(f"/workspace/ml_approx/data/alexnet_cifar10/aA_ab_l2_{args.method}.npy", l2s)
        np.save(f"/workspace/ml_approx/data/alexnet_cifar10/aA_ab_cossim_{args.method}.npy", cossims)
    elif args.metric == "acc":
        accs, losses = actual_Ab_accs_and_losses(func)
        np.save(f"/workspace/ml_approx/data/alexnet_cifar10/aA_ab_acc_{args.method}.npy", accs)
        np.save(f"/workspace/ml_approx/data/alexnet_cifar10/aA_ab_loss_{args.method}.npy", losses)
    else:
        raise ValueError(f"Unknown metric: {args.metric}")
    
    print("Done!")