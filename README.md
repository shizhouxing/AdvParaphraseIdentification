# Adversarial Paraphrase Identification

This is the code for our paper "Robustness to Modification with Shared Words in Paraphrase Identification" in Findings of EMNLP 2020.
The paper studies the adversarial robustness problem in paraphrase identification models.
On this task, we propose a method for generating adversarial examples, using adversarial modification with *shared words* (words shared by both two sentences in paraphrase identification). 
Our method has two stages to query the target model and find optimal positions for word substitution and optimal substitution words respectively, where the candidate substitution words are generated using the BERT masked language model. 
See examples below:

## Examples

### Label: Positive

|          | Sentence P                                  | Sentence Q                                            |
| -------- | ------------------------------------------- | ----------------------------------------------------- |
| Original | What is ultimate **purpose** of **life** ?  | What is the **purpose** of **life** , if not money ?  |
| Modified | What is ultimate **measure** of **value**? | What is the **measure** of **value** , if not money ? |

### Label: Negative

|          | Sentence P                            | Sentence Q                                    |
| -------- | ------------------------------------- | --------------------------------------------- |
| Original | How can I get my **Gmail account** back ? | What is the best **school management** software ? |
| Modified | How can I get my **credit score** back ?  | What is the best **credit score** software ?      |





## Dependencies

* Python 3.7

* Download and extract [datasets](https://huggingface.co/datasets/zhouxingshi/AdvParaphraseIdentification/blob/main/data.zip) to `data/`. The datasets include [QQP](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) and [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398).

* Download and extract [Glove word vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) to `data/`.

* Download and extract [pre-trained BERT base model](https://huggingface.co/datasets/zhouxingshi/AdvParaphraseIdentification/blob/main/bert_base.zip) to this directory.

* Install PyTorch 1.1.0 (using conda as an example):

```bash
conda create --name para python=3.7
conda activate para
conda install pytorch==1.1.0 cudatoolkit=10.0 -c pytorch
```

* Install other Python libraries:

```bash
pip install -r requirements.txt
```

## Normally Pre-train the Models

To normally pre-train the models using clean examples:

* For BERT ([original](https://github.com/google-research/bert), [PyTorch version](https://github.com/huggingface/transformers)):

```bash
cp -r bert_base bert_qqp|bert_mrpc
python main.py --target bert --data qqp|mrpc --dir bert_qqp|bert_mrpc --pre_train
```

* For [BiMPM](https://github.com/zhiguowang/BiMPM) and [DIIN](https://github.com/YichenGong/Densely-Interactive-Inference-Network):

```bash
python main.py --target bimpm|diin --data qqp|mrpc --dir bimpm_qqp|bimpm_mrpc|diin_qqp|diin_mrpc --pre_train
```

We also release our pre-trained BERT models for paraphrase identification on QQP and MRPC. You may directly use them, by downloading and extracting the model checkpoints ([QQP](https://drive.google.com/file/d/1NdnDttXKNFvQy0vDk_KPiGBwmmYd3KEJ/view?usp=sharing) and [MRPC](https://drive.google.com/file/d/1YWlt4AFyi5aG_v7eJAtr1KWZq6xJ7foK/view?usp=sharing)) to the current directory.

## Generate Adversarial Examples

Generate adversarial examples to evaluate the robustness of the models to our proposed adversarial modifications:

```
python main.py --data qqp|mrpc --target bert|bimpm|diin --dir bert_qqp|bert_mrpc|bimpm_qqp|bimpm_mrpc|diin_qqp|diin_mrpc --adv_generate --adv_file OUTPUT_FILE
```

## Adversarial Training

Run adversarial training to improve robustness:

```
python main.py --data qqp|mrpc --target bert|bimpm|diin --dir bert_qqp|bert_mrpc|bimpm_qqp|bimpm_mrpc|diin_qqp|diin_mrpc --adv_train
```

Do early stopping when the training converges.

## Citation

Please kindly cite our paper:

Zhouxing Shi and Minlie Huang. [Robustness to Modification with Shared Words in Paraphrase Identification](https://aclanthology.org/2020.findings-emnlp.16.pdf). In Findings of EMNLP 2020.

```
@inproceedings{shi2020robustness,
  title={Robustness to Modification with Shared Words in Paraphrase Identification},
  author={Shi, Zhouxing and Huang, Minlie},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={164--171},
  year={2020}
}
```
