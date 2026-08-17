from transformers import AutoTokenizer
from src.model import BertClassificationModel
from src.utils import *
from pathlib import Path
from src.configloader import ConfigLoader

class TextPredict:
    def __init__(self,cfg):
        self.device = get_device(cfg.runtime.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_save_dir)
        self.model = BertClassificationModel(cfg).to(self.device)
        self.label_info = load_label_map(f"{cfg.model.model_save_dir}/label_map.json")
        self.class_names = self.label_info["class_names"]
        self.max_len = cfg.model.max_len

    def predict(self, text):
        self.model.load_state_dict(torch.load(f"{cfg.model.model_save_dir}/best_bert.bin", weights_only=True))
        input = self.tokenizer(text, max_length= self.max_len,
                               truncation=True, padding="max_length",
                               return_tensors="pt")
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)

        with torch.no_grad():
            _, logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred_idx = int(torch.argmax(logits, dim=1).cpu().item())

        pred_label = id2name[label2id[pred_idx]]
        return pred_label

if __name__ == "__main__":
    config_json = Path("../config/config.json")
    cfg = ConfigLoader(config_json)
    predictor = TextPredict(cfg)
    print("输入新闻文本分类，输入q退出\n")

    while True:
        content = input("请输入新闻标题：")
        if content.strip() == "q":
            print("程序退出")
            break
        if not content.strip():
            print("内容为空")
            continue

        label = predictor.predict(content)
        print(f"预测分类：{label}\n")