# Auxilary Gaze Task for Financial Text Sentiment Analysis

This project is for SJTU 39th student research project.  
Contributer: Yuan Fang, YinYi Min, Wenbo Li, Yuqi Zhou



## Quick Start

```
conda create -n=attention python=3.8  
conda activate attention  
pip install -r requirements.txt
cd src
python gaze.py
```


## Adjust for your own recipe
```
python gaze.py \
    --random_seed=22 \
    --dataset_name="Round2" \
    --model_name="bert-base-uncased" \
    --attack_method="default"
```

You can write your own dataset class and download the corresponding data under /src/gaze_data/ .

Model class includes but is not limited to:
* Bert
* Other variants of Bert that has been implemented in $transformers$


Attack Method includes but is not limited to:  
* default
* bert_attack

