import random
import argparse
import torch
import os
import json
from pathlib import Path
from configloader import ConfigLoader
import sklearn.metrics as skm

id2name = {
    100: "民生",
    101: "文化",
    102: "娱乐",
    103: "体育",
    104: "财经",
    106: "房产",
    107: "汽车",
    108: "教育",
    109: "科技",
    110: "军事",
    112: "旅游",
    113: "国际",
    114: "证券",
    115: "农业",
    116: "电竞"
}
id2label = {
    100: 0,
    101: 1,
    102: 2,
    103: 3,
    104: 4,
    106: 5,
    107: 6,
    108: 7,
    109: 8,
    110: 9,
    112: 10,
    113: 11,
    114: 12,
    115: 13,
    116: 14
}
label2id = {v: k for k, v in id2label.items()}
class_names = list(id2name.values())
NUM_CLASSES = 15


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def save_label_map(save_path):
    """保存标签映射json，放到模型目录，推理读取"""
    map_data = {
        "id2name": id2name,
        "id2idx": id2label,
        "idx2id": label2id,
        "class_names": class_names,
        "num_classes": NUM_CLASSES
    }
    with open(save_path, "w", encoding="utf‑8") as f:
        json.dump(map_data, f, ensure_ascii=False, indent=2)


def load_label_map(json_path):
    with open(json_path, "r", encoding="utf‑8") as f:
        return json.load(f) # 返回字典

def get_metrics(true, pred,classes_num):

    true = true.squeeze()
    pred = pred.squeeze()


    total = len(true)
    correct = 0
    tp = [0] * classes_num  # 真实是A    预测是A
    fp = [0] * classes_num  # 真实不是A  预测是A
    fn = [0] * classes_num  # 真实是A    预测不是A

    for true_label, pred_label in zip(true, pred):
        if true_label == pred_label:
            correct += 1
            tp[true_label] += 1  # 真实=预测
        else:
            fn[true_label] += 1  # 真实是true_label，但预测错了，被预测错的加
            fp[pred_label] += 1  # 预测成pred_label，预测成的加

    acc = correct / total if total != 0 else 0

    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(classes_num):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) != 0 else 0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0
        precision_list.append(p)
        recall_list.append(r)

    for p, r in zip(precision_list, recall_list):
        f = 2 * p * r / (p + r) if p + r != 0 else 0
        f1_list.append(f)

    precision = sum(precision_list) / classes_num
    recall = sum(recall_list) / classes_num
    f1 = sum(f1_list) / classes_num


    return {
        "acc": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

class EarlyStopping:
    def __init__(self, cfg):
        self.patience = cfg.train.early_stop_patience
        self.counter = 0
        self.loss = None
        self.early_stop = False

    def __call__(self, cur_loss: float):
        if self.loss is None:
            self.loss = cur_loss
        elif cur_loss >= self.loss:
            self.counter += 1
            print(f"[早停计数器] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.loss = cur_loss
            self.counter = 0

def set_seed(seed=42):
    # Dataloader shuffle
    random.seed(seed)
    # dropout cpu、参数初始化
    torch.manual_seed(seed)
    # dropout gpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 关闭cuda非确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # 测试 是否正确
    true = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    pred = torch.tensor([0,0,2,3,5,5,6,1,8,9])
    t = true.numpy()
    p = pred.numpy()
    acc_sk = skm.accuracy_score(t, p)
    pr_sk = skm.precision_score(t, p, average="macro", zero_division=0)
    re_sk = skm.recall_score(t, p, average="macro", zero_division=0)
    f1_sk = skm.f1_score(t, p, average="macro", zero_division=0)

    print("acc:",acc_sk)
    print("pr:",pr_sk)
    print("re:",re_sk)
    print("f1:",f1_sk)
    print(get_metrics(true,pred,10))
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
    early_stop = EarlyStopping(cfg)

    list = [10,9,8,6,7,5,7,7,4,4,3]
    for i in list :
        print(i)
        early_stop(i)
        print("--------")
        if early_stop.early_stop:
            break