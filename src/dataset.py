import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from configloader import ConfigLoader
from utils import id2label
from pathlib import Path


# 重写 __init__、__len__、__getitem__
class NewsDataset(Dataset):
    def __init__(self, max_len):
        self.samples = []
        self.max_len = max_len

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("_!_")
                id = parts[1]  # 新闻类别号
                text = parts[3]  # 新闻题目
                self.samples.append({"id": id, "text": text})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        text = item["text"]
        id = int(item["id"])
        labels = torch.tensor(id2label[id], dtype=torch.long)

        return{
            "text": text,
            "labels": labels
        }

if __name__ == "__main__":
    cfg = ConfigLoader(Path("../config/config.json"))

    file_path = cfg.data.train_path
    max_len = cfg.model.max_len
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    dataset = NewsDataset(tokenizer, max_len)
    dataset.load(file_path)
    print(dataset)
    print(dataset[0])
    print(dataset[0]["input_ids"].shape)
    print(dataset[0]["labels"])