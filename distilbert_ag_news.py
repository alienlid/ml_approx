import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
from tqdm import tqdm
import approx_methods
import argparse

# _____________________________________________
# l2 error
def actual_Ab_l2_dists(f):
    l2_dists = []
    model.eval()

    for c in tqdm(cs):
        weight_approx = f(weight, c)
        diff = torch.from_numpy(weight - weight_approx).cuda()
        l2_dist = 0
        for batch in test_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            with torch.no_grad():
                b = model.distilbert(input_ids, attention_mask=attention_mask)
                b = b.last_hidden_state[:, 0, :]
                error = b @ diff.T
                l2_dist += torch.sum(torch.norm(error, dim=1) / torch.norm(b, dim=1))
        
        l2_dist /= len(test_loader.dataset) * np.linalg.norm(weight)
        l2_dists.append(l2_dist.item())
        tqdm.write(f"method: {args.method}, c: {c}, l2_dist: {l2_dist.item()}")

    return l2_dists

# _____________________________________________
# accuracy
def actual_Ab_accs(f):
    accs = []
    model.eval()
    for c in tqdm(cs):
        weight_approx = f(weight, c)
        model.pre_classifier.weight.data = torch.from_numpy(weight_approx).cuda()
        metric = evaluate.load("accuracy")
        for batch in test_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

            metric.add_batch(predictions=predictions.cpu(), references=labels.cpu())

        result = metric.compute()
        accs.append(result["accuracy"])
        tqdm.write(f"method: {args.method}, c: {c}, accuracy: {result['accuracy']}")
    return accs

# _____________________________________________

if __name__ == "__main__":
    # get method from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    args = parser.parse_args()
    func = getattr(approx_methods, args.method)
    print(f"method: {args.method}, metric: {args.metric}")

    # load dataset and model
    dataset = load_dataset("ag_news")
    dataset = dataset.rename_column("label", "labels")

    model_name = "textattack/distilbert-base-uncased-ag-news"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
    weight = model.pre_classifier.weight.data.cpu().numpy()

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    encoded = dataset["test"].map(preprocess, batched=True)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    test_loader = DataLoader(encoded, batch_size=64)

    # compression ratios
    # cs = np.concat((
    #     np.arange(0, 128, 8),
    #     np.arange(128, 192, 4),
    #     np.arange(192, 224, 2),
    #     np.arange(224, 257, 1),
    # )) / 256
    cs = np.linspace(0, 1, 769)

    if args.metric == "l2":
        l2_dists = actual_Ab_l2_dists(func)
        np.save(f"/workspace/ml_approx/data/distilbert_ag_news/aA_ab_l2_{args.method}.npy", l2_dists)
    elif args.metric == "acc":
        accs = actual_Ab_accs(func)
        np.save(f"/workspace/ml_approx/data/distilbert_ag_news/aA_ab_acc_{args.method}.npy", accs)
    else: 
        raise ValueError(f"Unknown metric: {args.metric}")
    
    print("Done!")