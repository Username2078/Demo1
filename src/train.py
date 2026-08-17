from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import swanlab
from dataset import NewsDataset
from src.model import BertClassificationModel
from utils import *
from pathlib import Path
from configloader import ConfigLoader
import argparse



def evaluate(model, test_dataloader,num_classes ,device):

    model.eval()
    true_list = []
    pred_list = []
    loss_total = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_total += loss.item()
            preds = torch.argmax(logits, dim=1)
            true_list.append(labels.cpu())
            pred_list.append(preds.cpu())

    true = torch.cat(true_list, dim=0)
    pred = torch.cat(pred_list, dim=0)

    loss_avg = loss_total / len(test_dataloader)
    metrics = get_metrics(true, pred, num_classes)
    metrics["loss"] = loss_avg
    return metrics

def train(cfg):
    set_seed(cfg.train.seed)
    train_path = cfg.data.train_path
    eval_path = cfg.data.eval_path
    model_save_dir = cfg.model.model_save_dir
    tokenizer_save_dir = cfg.model.tokenizer_save_dir
    max_len = cfg.model.max_len
    num_classes = cfg.model.num_classes
    device = get_device(cfg.runtime.device)
    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs
    lr = cfg.train.lr

    make_dir(model_save_dir)
    save_label_map(f"{model_save_dir}/label_map.json")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

    train_dataset = NewsDataset(tokenizer, max_len)
    train_dataset.load(train_path)
    eval_dataset = NewsDataset(tokenizer, max_len)
    eval_dataset.load(eval_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = BertClassificationModel(cfg)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    swanlab.init(
        project="Demo_1",
        config=cfg.config_dict,
        experiment_name=cfg.swanlab.experiment_name,
    )
    max_acc = 0
    early_stop = EarlyStopping(cfg)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_avg = total_loss / len(train_dataloader)
        print(f"===第{epoch}次，损失：{loss_avg:.4f}===")

        metrics = evaluate(model, test_dataloader, num_classes,device)
        print(f"验证损失:{metrics['loss']:.4f} 准确率:{metrics['acc']:.4f} f1:{metrics['f1']:.4f}")

        swanlab.log({
            "train_loss": loss_avg,
            "acc": metrics["acc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        }, step = epoch)

        if metrics['acc'] > max_acc:
            print(f"准确率{metrics['acc']}大于{max_acc}")
            max_acc = metrics['acc']

            save_file = os.path.join(cfg.model.model_save_dir, "best_bert.bin")
            torch.save(model.state_dict(), save_file)
            tokenizer.save_pretrained(tokenizer_save_dir)

        # 早停
        early_stop(metrics['loss'])
        if early_stop.early_stop:
            print(f"连续{early_stop.patience}损失未下降，停止训练")
            break
    swanlab.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT新闻分类训练")
    parser.add_argument("--config_path", type=str, default="../config/config.json", help="配置json文件路径")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--experiment_name", type=str)
    args = parser.parse_args()

    config_json = Path(args.config_path)
    cfg = ConfigLoader(config_json)

    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.max_len is not None:
        cfg.model.max_len = args.max_len
    if args.num_classes is not None:
        cfg.model.num_classes = args.num_classes
    if args.experiment_name is not None:
        cfg.swanlab.experiment_name = args.experiment_name

    train(cfg)
