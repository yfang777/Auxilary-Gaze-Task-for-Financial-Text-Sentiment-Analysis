import click
from typing import Dict

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from bert_attack import *

def get_model(pretrained_model_name: str):
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
    
    config = AutoConfig.from_pretrained(pretrained_model_name)
    id2label = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    label2id = {y:x for x,y in id2label.items()}
    config.label2id = label2id
    config.id2label = id2label
    config._num_labels = len(label2id)

    bert = AutoModel.from_pretrained(pretrained_model_name)
    sequence_classifier = AutoModelForSequenceClassification.from_config(config)
    sequence_classifier.bert = bert
    
    return sequence_classifier

def get_Dataset(dataset_name: str):
    from gazedataset import Round2_Dataset
    
    if dataset_name == "Round2":
        return Round2_Dataset()
    else:
        raise NotImplementedError

def tokenize_gaze(
    seq_detail: Dict,
    tokenizer,
    max_length = 32):
    
    seq_detail_tokenized = {}
    seq_words = []
    seq_input_ids = []
    seq_attention_mask = []
    seq_nFixation= []
    
    for word_detail in seq_detail:
        seq_words.append(word_detail["word"])
        
        if not seq_input_ids:
            seq_input_ids.append(101)
            seq_attention_mask.append(1)
            seq_nFixation.append(0)
        
        # Truncation
        word_tokenized = tokenizer(word_detail["word"])
        for input_ids in word_tokenized["input_ids"][1:-1]:
            if len(seq_input_ids) < max_length - 2:
                seq_input_ids.append(input_ids)
                seq_nFixation.append(word_detail["nFixation"])

        for attention_mask in word_tokenized["attention_mask"][1:-1]:
            if len(seq_attention_mask) < max_length - 2:
                seq_attention_mask.append(attention_mask)
    

    # Padding
    for _ in range(len(seq_input_ids), max_length - 1):
        seq_input_ids.append(0)
        seq_attention_mask.append(0)
        seq_nFixation.append(0)
    
    seq_input_ids.append(0)
    seq_attention_mask.append(0)
    seq_nFixation.append(0)


    if len(seq_input_ids) > max_length:
        print(seq_input_ids)
        raise ValueError
     
    seq_detail_tokenized["words"] = seq_words
    seq_detail_tokenized["input_ids"] = seq_input_ids
    seq_detail_tokenized["attention_mask"] = seq_attention_mask
    seq_detail_tokenized["nFixation"] = seq_nFixation
    return seq_detail_tokenized

def tokenize_batch(
    batch_seq: Dict,
    tokenizer,
    ):
    
    batch_tokenized = {}
    batch_text = []
    
    batch_sentiment = []
    batch_input_ids = []
    batch_attention_mask = []
    batch_nFixation = []

    for seq in batch_seq:
        batch_text.append(seq["text"])
        batch_sentiment.append(seq["label"])
        seq_detail_tokenized = tokenize_gaze(seq_detail=seq["detail"], tokenizer=tokenizer)
        batch_input_ids.append(seq_detail_tokenized["input_ids"])
        batch_attention_mask.append(seq_detail_tokenized["attention_mask"])
        batch_nFixation.append(seq_detail_tokenized["nFixation"])

    batch_tokenized["text"] = batch_text
    # batch_tokenized["text_attack"] = batch_text
    batch_tokenized["input_ids"] = torch.Tensor(batch_input_ids).int()
    batch_tokenized["attention_mask"] = torch.Tensor(batch_attention_mask).int()
    
    batch_tokenized["sentiment_label"] = torch.Tensor(batch_sentiment).type(torch.LongTensor) 
    batch_tokenized["nFixation"] = torch.Tensor(batch_nFixation).type(torch.LongTensor) 

    return batch_tokenized

def get_dataloader(
        dataset,
        tokenizer,
        seed,
        split_ratio: list =[0.7, 0.1, 0.2],
        train_batch: int = 16,
        ):
    
    
    train_size = int(split_ratio[0] * len(dataset))
    valid_size = int(split_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(seed))
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, collate_fn= lambda x: tokenize_batch(batch_seq=x, tokenizer=tokenizer))
    valid_dateloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn= lambda x: tokenize_batch(batch_seq=x, tokenizer=tokenizer))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tokenize_batch(batch_seq=x, tokenizer=tokenizer))

    return train_dataloader, valid_dateloader, test_dataloader

def train(
    model, 
    train_dataloader, 
    logger, 
    epoch: int,
    learning_rate: float = 1e-4,
    normalize_rate: float = 10,
    device: str = "cuda"):
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_sentiment_fn = nn.CrossEntropyLoss()
    loss_nFixation_fn = nn.PoissonNLLLoss()
    
    loss_l = []
    loss_sentiment_l = []
    loss_nFixation_l = []

    for i in range(epoch):
        for batch_tokenized in train_dataloader:
            text = batch_tokenized["text"]    # list: N * str
            input_ids = batch_tokenized["input_ids"].to(device)
            attention_mask = batch_tokenized["attention_mask"].to(device)
            nFixation = batch_tokenized["nFixation"].to(device)
            sentiment_label = batch_tokenized["sentiment_label"].to(device)

            optimizer.zero_grad()
            sentiment_output, nFixation_output = model(input_ids, attention_mask=attention_mask)
            
            loss_sentiment = loss_sentiment_fn(sentiment_output, sentiment_label)
            loss_nFixation = loss_nFixation_fn(nFixation_output, nFixation)

            loss = loss_sentiment + normalize_rate * loss_nFixation
            
            loss_l.append(loss.detach().cpu().numpy())
            loss_sentiment_l.append(loss_sentiment.detach().cpu().numpy())
            loss_nFixation_l.append(loss_nFixation.detach().cpu().numpy())
            
            loss.backward()
            optimizer.step()

        logger.log_str(f"Train Epoch {i}: loss {np.mean(loss_l)}")
        logger.log_str(f"Train Epoch {i}: loss_sentiment: {np.mean(loss_sentiment_l)}")
        logger.log_str(f"Train Epoch {i}: loss_nFixation: {np.mean(loss_nFixation_l)}")
        
    return np.sum(loss_l)

def test(model,
          test_dataloader,
          logger,
          device: str="cuda"):

    model.eval()

    y_pred_list = []
    acc_count = 0
    
    for batch_tokenized in test_dataloader:
        text = batch_tokenized["text"]    # list: N * str
        input_ids = batch_tokenized["input_ids"].to(device)
        attention_mask = batch_tokenized["attention_mask"].to(device)
        nFixation = batch_tokenized["nFixation"].to(device)
        sentiment_label = batch_tokenized["sentiment_label"].to(device)

        sentiment_output, nFixation_output = model(input_ids, attention_mask=attention_mask)

        _, y_pred_tags = torch.max(sentiment_output, dim = 1)
        if y_pred_tags == sentiment_label:
            acc_count += 1
        
        y_pred_list.append(y_pred_tags.cpu().numpy())

    logger.log_str(f"Test Accuracy { acc_count / len(y_pred_list)}")

def get_attack_model(
    model,
    pretrained_model_name = "bert-base-uncased",
    ):
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
    
    config = AutoConfig.from_pretrained(pretrained_model_name)
    id2label = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    label2id = {y:x for x,y in id2label.items()}
    config.label2id = label2id
    config.id2label = id2label
    config._num_labels = len(label2id)

    sentence_classifier = AutoModelForSequenceClassification.from_config(config)
    sentence_classifier.bert = model.base_model
    sentence_classifier.classifier = model.sentiment
    
    return sentence_classifier

def attacktest(classifier,
            attack,
            dataloader,
            logger):
    robust_count = 0
    total_count = 0
    for batch_tokenized in dataloader:
        text = str(batch_tokenized["text"][0])
        original_label = int(batch_tokenized["sentiment_label"])
        attack_result = attack.attack(text, original_label)
        print(attack_result)
        print("original_label:", original_label)
        if isinstance(attack_result, SkippedAttackResult):
            continue
        elif isinstance(attack_result, SuccessfulAttackResult):
            total_count += 1
            
        elif isinstance(attack_result, FailedAttackResult):
            robust_count += 1
            total_count += 1
        else:
            raise ValueError(f"Unrecognized goal status {attack_result.goal_status}")
            
    logger.log_str(f"Attack Robustness Rate { robust_count / total_count}")


@click.command()
@click.option("--random_seed", default=22)
@click.option("--normalize_rate", default=10)
@click.option("--dataset_name", default="Round2")
@click.option("--model_name", default="bert-base-uncased")
@click.option("--attack_method", default="default")
def main(
    random_seed: int,
    normalize_rate: float,
    dataset_name: str,
    model_name: str,
    attack_method: str
    ):
    from utils import set_global_seed, Logger, set_logger, EarlyStopper
    from transformers import AutoTokenizer
    
    set_global_seed(random_seed)
    dataset = get_Dataset(dataset_name)
    device = "cuda" if torch.cuda.is_available else "cpu"
    
    from gazemodel import Bert_sentiment_gaze 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Bert_sentiment_gaze(model_name)
    model.to(device)
    logger = Logger(log_path=".logs", env_name="Attention", seed=random_seed)
    set_logger(logger)
    logger.log_str(f"Random Seed: {random_seed}")
    logger.log_str(f"Normalize_Rate: {normalize_rate}")
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(dataset=dataset, tokenizer=tokenizer, split_ratio=[0.7, 0, 0.3], seed=random_seed)
    train(model=model, train_dataloader=train_dataloader, logger=logger, epoch=10, normalize_rate=normalize_rate)
    test(model=model, test_dataloader=test_dataloader, logger=logger)
    
    
    classifier_attack = get_attack_model(model=model)
    attack = get_Attack(classifier_attack, tokenizer, attack_method=attack_method)
    attacktest(classifier=classifier_attack, attack=attack, dataloader=test_dataloader, logger=logger)
    
if __name__ == "__main__":
    main()