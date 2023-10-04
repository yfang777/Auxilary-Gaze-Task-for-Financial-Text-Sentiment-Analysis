from transformers import BertPreTrainedModel, AutoModel, BertModel
from torch import nn



# class Distilbert_sentiment_gaze(nn.Module):
#     def __init__(self, pretrained_model_name, dropout_rate=0.1):
#         super().__init__()
#         self.sentiment = nn.Linear(self.base_model.config.hidden_size, 3)
#         self.nFixation= nn.Linear(self.base_model.config.hidden_size, 1)
#         self.sentiment_dropout = nn.Dropout(p=dropout_rate)
#         self.nFixation_dropout = nn.Dropout(p=dropout_rate)
        

#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         output1 = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["last_hidden_state"] # Batch * len of Seq * Embedding
#         output2 = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["pooler_output"] # Batch     
#         sentiment_output = self.sentiment(self.sentiment_dropout(output2))   # Batch * 3
#         nFixation_output = self.nFixation(self.nFixation_dropout(output1)).squeeze(-1)  # Batch * len of Seq
#         return sentiment_output, nFixation_output

class Bert_sentiment_gaze(nn.Module):
    def __init__(self, pretrained_model_name, dropout_rate=0.1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        self.sentiment = nn.Linear(self.base_model.config.hidden_size, 3)
        self.nFixation= nn.Linear(self.base_model.config.hidden_size, 1)
        self.sentiment_dropout = nn.Dropout(p=dropout_rate)
        self.nFixation_dropout = nn.Dropout(p=dropout_rate)
        

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output1 = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["last_hidden_state"] # Batch * len of Seq * Embedding
        output2 = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["pooler_output"] # Batch     
        sentiment_output = self.sentiment(self.sentiment_dropout(output2))   # Batch * 3
        nFixation_output = self.nFixation(self.nFixation_dropout(output1)).squeeze(-1)  # Batch * len of Seq
        
        return sentiment_output, nFixation_output

