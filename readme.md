# 此版本为尚未完结版本，直接使用query set得到的边界loss和类型loss



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

### Step1: Prepare BERT embedding:
- Download the pytorch bert model, or convert tensorflow param by yourself as follow:
```bash
export BERT_BASE_DIR=/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch
  $BERT_BASE_DIR/bert_model.ckpt
  $BERT_BASE_DIR/bert_config.json
  $BERT_BASE_DIR/pytorch_model.bin
```
- Set BERT path in the file `./scripts/run_SpanProtonet.sh` to your setting:
```bash
bert_base_uncased=/your_dir/uncased_L-12_H-768_A-12/
bert_base_uncased_vocab=/your_dir/uncased_L-12_H-768_A-12/vocab.txt
```


### Step2: Prepare data
- Set test, train, dev data file path in `./scripts/run_SpanProtonet.sh` to your setting.
  
> For simplicity, your only need to set the root path for data as follow:
```bash
base_data_dir=/your_dir/data/
```

### Step3: Train and test the main model
- Build a folder to collect running log
```bash
mkdir result
```

- Execute cross-evaluation script with two params: -[gpu id] -[dataset name]

##### Example for 1-shot Snips:
```bash
source ./scripts/run_SpanProtonet.sh 0
```
##### Example for 1-shot NER:
```bash
source ./scripts/run_SpanProtonet.sh 0
```

> To run 5-shots experiments, use `./scripts/run_SpanProtonet_5.sh`


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
    - Boundary Detection Module (`SpanDetector.py`): modules that detect entity boundary
    - Boundary Output Module (`crf.py`): boundary detection output layer with conditional random field 
    - Span Classification Module(`span_classification_base.py`): modules that classify the span by metric learning
    - Output Module (`span_entity_labeler.py`): output layer with normal mlp.
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



--do_debug --do_train --do_predict --delete_checkpoin --load_feature --train_path processed_data/v3/train_2.json --dev_path processed_data/v3/test_2.json --test_path processed_data/v3/test_2.json --output_dir outputs_models/model/2 --bert_path /home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12 --bert_vocab /home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12/vocab.txt --train_batch_size 2 --cpt_per_epoch 2 --gradient_accumulation_steps 1 --num_train_epochs -1 --warmup_epoch 0 --test_batch_size 2 --context_emb sep_bert --label_reps sep --emission proto --similarity dot --ems_scale_r 0.01 --decoder sms


#
export BERT_BASE_DIR=/home/cike/hui/Pre-BERTs/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch $BERT_BASE_DIR/bert_model.ckpt $BERT_BASE_DIR/bert_config.json $BERT_BASE_DIR/pytorch_model.bin



##积累
torch.gather()
unsqueeze()
squeeze_()
.expand()