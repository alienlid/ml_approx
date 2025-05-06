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
        for batch in test_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            with torch.no_grad():
                b = model.distilbert(input_ids, attention_mask=attention_mask)
                b = b.last_hidden_state[:, 0, :]
                error = b @ (weight_cuda - weight_approx).T
                l2 += torch.sum(torch.norm(error, dim=1) / torch.norm(b, dim=1))
                cossim += torch.sum(torch.nn.functional.cosine_similarity(
                    b @ weight_cuda.T,
                    b @ weight_approx.T,
                    dim=1,
                ))
        
        l2 /= len(test_loader.dataset) * np.linalg.norm(weight)
        cossim /= len(test_loader.dataset)
        l2s.append(l2.item())
        cossims.append(cossim.item())
        tqdm.write(f"method: {args.method}, c: {c}, l2: {l2.item()}, cossim: {cossim.item()}")

    return l2s, cossims

# _____________________________________________
# accuracy
def actual_Ab_accs_and_losses(f):
    accs = []
    losses = []
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    for c in tqdm(cs):
        weight_approx = f(weight, c)
        model.pre_classifier.weight.data = torch.from_numpy(weight_approx).cuda()
        metric = evaluate.load("accuracy")

        total_loss = 0.0

        for batch in test_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

            metric.add_batch(predictions=predictions.cpu(), references=labels.cpu())

        result = metric.compute()
        accs.append(result["accuracy"])
        losses.append(total_loss / len(test_loader.dataset))
        tqdm.write(f"method: {args.method}, c: {c}, acc: {accs[-1]}, loss: {losses[-1]}")
    return accs, losses

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
    weight_cuda = torch.from_numpy(weight).cuda() # dumb code lmao

    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    encoded = dataset["test"].map(preprocess, batched=True)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    test_loader = DataLoader(encoded, batch_size=64)

    # compression ratios
    cs = np.linspace(0, 1, 769)

    if args.metric == "l2":
        l2s, cossims = actual_Ab_l2s_and_cossims(func)
        # np.save(f"/workspace/ml_approx/data/distilbert_ag_news/aA_ab_l2_{args.method}.npy", l2s)
        np.save(f"/workspace/ml_approx/data/distilbert_ag_news/aA_ab_cossim_{args.method}.npy", cossims)
    elif args.metric == "acc":
        accs, losses = actual_Ab_accs_and_losses(func)
        np.save(f"/home/ddl/ml_approx/data/distilbert_ag_news/aA_ab_acc_{args.method}.npy", accs)
        np.save(f"/home/ddl/ml_approx/data/distilbert_ag_news/aA_ab_loss_{args.method}.npy", losses)
    else: 
        raise ValueError(f"Unknown metric: {args.metric}")
    
    print("Done!")
