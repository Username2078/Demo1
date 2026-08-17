from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import NewsDataset
from src.model import BertClassificationModel
from src.utils import *
from src.configloader import ConfigLoader
from pathlib import Path


def test(cfg):
    device = get_device(cfg.runtime.device)
    model = BertClassificationModel(cfg).to(device)
    model.load_state_dict(torch.load(f"{cfg.model.model_save_dir}/best_bert.bin", weights_only=True))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_save_dir)
    label_info = load_label_map(f"{cfg.model.model_save_dir}/label_map.json")
    num_classes = cfg.model.num_classes
    max_len = cfg.model.max_len
    test_data_path =  cfg.data.test_path

    test_dataset = NewsDataset(tokenizer, max_len)
    test_dataset.load(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    total_loss = 0.0
    total_acc = 0
    true_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            true_list.append(labels.cpu())
            pred_list.append(preds.cpu())
            pred = torch.argmax(logits, dim = 1)
            total_acc += (pred == labels).sum().item()
            true = torch.cat(true_list, dim=0)
            pred = torch.cat(pred_list, dim=0)

        metrics = get_metrics(true, pred, num_classes)
        print("\n========测试集最终结果========")
        print(f"测试损失:{total_loss/len(test_dataloader):.4f}")
        print(f"准确率: {metrics['acc']}")
        print(f"precision查准率: {metrics['precision']}")
        print(f"recall召回率: {metrics['recall']}")
        print(f"f1分数: {metrics['f1']}")


if __name__ == "__main__":
    config_json = Path("../config/config.json")
    cfg = ConfigLoader(config_json)
    test(cfg)