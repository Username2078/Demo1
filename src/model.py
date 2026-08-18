from torch import nn
from transformers import BertModel


class BertClassificationModel(nn.Module):
    def __init__(self, cfg):
        super(BertClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(cfg.model.origin_model_dir)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, cfg.model.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bert_ouputs = self.bert(input_ids, attention_mask = attention_mask,
                           token_type_ids=token_type_ids)

        outputs = self.dropout(bert_ouputs.pooler_output)
        logits = self.linear(outputs)
        loss = None

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss,logits
        