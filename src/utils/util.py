import os
import numpy as np
import random
import torch
import textstat

def get_subfiles(directory):
    subdirectories = []
    subfiles = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            subdirectories.append(os.path.join(root, dir))
        for file in files:
            subfiles.append(os.path.join(root, file))
    return subdirectories, subfiles

def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logger(logger_ent):
    global logger
    logger = logger_ent

def cal_flesch_reading_ease(seq_batch: list, Normalize=True):
    l = []
    for seq in seq_batch:
        if Normalize:
            l.append(textstat.flesch_reading_ease(seq) / 10)
        else:
            l.append(textstat.flesch_reading_ease(seq))
    return l

def cal_flesch_kincaid_grade(seq_batch: list):
    l = []
    for seq in seq_batch:
        l.append(textstat.flesch_kincaid_grade(seq))
    return l

def save_model(PATH, model, optimizer):
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

