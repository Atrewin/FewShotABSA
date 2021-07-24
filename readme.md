
# Fine-grained Few-shot 

## Get Started

### Requirement
```
python >= 3.6
pytorch >= 0.4.1
pytorch_pretrained_bert >= 0.6.1
allennlp >= 0.8.2
torchnlp
```  


(simply, can copy pre-bert and dataset from 234/hui/Projects/FGFew~~, then go to step3)  

### Step1: Prepare BERT embedding:
- Download the pytorch bert model, or convert tensorflow param by yourself as follow:
```bash
export BERT_BASE_DIR=/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch
  $BERT_BASE_DIR/bert_model.ckpt
  $BERT_BASE_DIR/bert_config.json
  $BERT_BASE_DIR/pytorch_model.bin
```
- Set BERT path in the file to `utils/opt.py` your setting:


### Step2: Prepare data

> For simplicity, your only need to set the root path for data as follow:
```bash
base_data_dir=/your_dir/data/
```

### Step3: Train and test the main model

## not CRF
> CUDA_VISIBLE_DEVICES=2 python main.py --do_debug --do_train --do_predict --delete_checkpoin --load_feature --train_path processed_data/v3/train_2.json --dev_path processed_data/v3/test_2.json --test_path processed_data/v3/test_2.json --output_dir outputs_models/model/2 --bert_path /home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12 --bert_vocab /home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12/vocab.txt --train_batch_size 2 --cpt_per_epoch 2 --gradient_accumulation_steps 1 --num_train_epochs -1 --warmup_epoch 0 --test_batch_size 2 --context_emb sep_bert --label_reps sep --emission proto --similarity dot --ems_scale_r 0.01 --decoder sms

## CRF
> CUDA_VISIBLE_DEVICES=2 python main.py --do_debug --do_train --eval_when_train --do_predict --delete_checkpoin --load_feature --train_path processed_data/v3/train_2.json --dev_path processed_data/v3/test_2.json --test_path processed_data/v3/test_2.json --output_dir outputs_models/model/2 --bert_path /home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12 --bert_vocab /home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12/vocab.txt --train_batch_size 2 --cpt_per_epoch 2 --gradient_accumulation_steps 1 --num_train_epochs -1 --warmup_epoch 0 --test_batch_size 2 --context_emb sep_bert --label_reps sep --emission proto --similarity dot --ems_scale_r 0.01 --decoder crf -t_scl learn -t_nm norm -lt_nm softmax -t_scl learn

- Execute cross-evaluation script with two params: -[gpu id] -[dataset name]



# wait to implement
##### Example for 1-shot Snips:
```bash

```
##### Example for 1-shot NER:
```bash

```

> To run 5-shots experiments, use 


## Project Architecture

### `Root`
- the project contains three main parts:
    - `models`: the neural network architectures
    - `scripts`: running scripts for cross evaluation
    - `utils`: auxiliary or tool function files
    - `main.py`: the entry file of the whole project

### `models`
- Main Model  
    - Sequence Labeler (`few_shot_seq_labeler.py`): a framework that integrates modules below to perform sequence labeling.
- Modules
    - Embedder Module (`context_embedder_base.py`): modules that provide embeddings.
    - Emsision Module (`emission_scorer_base.py`): modules that get tags logistics
    - Docoder Module (`torchcrf.CRF or seq_labeler.py`): modules that get loss for tags logistics
    - Scale Module (`scale_controller.py`): a toolkit for re-scale and normalize logits.

### `utils`

- `utils` contains assistance modules for:
    - data processing (`data_helper.py`, `preprocessor.py`), 
    - constructing model architecture (`model_helper.py`), 
    - controlling training process (`trainer.py`), 
    - controlling testing process (`tester.py`), 
    - controllable parameters definition (`opt.py`), 
    - device definition (`device_helper`) 
    - config (`config.py`).
    
##### few-shot/meta-episode style data example

```json
{
  "domain_name": [
    {  // episode
      "support": {  // support set
        "text": ["The", "speed", "is", "incredible", "and", "I", "am", "more", "than", "satisfied", "."],  // input sequence
        "labels": ["O", "T-POS", "O", "O", "O", "O", "O", "O", "O", "O", "O"],  // output sequence in sequence labeling task
        "sent": "POS"  // sentiment labels
      },
      "query": {  // query set
        "text": ["I", "am", "using", "the", "external", "speaker", "sound", "is", "good", "."],
        "labels": ["O", "O", "O", "O", "T-POS", "T-POS", "T-POS", "O", "O", "O"]
        "sent": "POS"  
      }
    },
    ...
  ],
  ...
}

```


# run in pycharm  


## 积累
torch.gather()
unsqueeze()
squeeze_()
.expand()
index_select()
argmax()

