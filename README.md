# DupBERT

DupBERT is a siamese neural network based on BERT helping to classify texts with similar meaning as duplicates.
The package utilises Catalyst (PyTorch framework for Deep Learning Research and Development) and can be easily run with different neural architectures.


## Installation
The current realisation is fully tested with python=3.9.

```
# Installation
git clone 
cd DupBERT 
pip install -r requirements.txt
```

## The artchitecture
The model architecture is as following:
* **BERT encoder.** Each text in triplet is passed to encoder where input_ids & attention mask are extracted.
* **Convolution layers**. Each encoded text's separately fed to the shared CNN layers and MaxPooled.
* **Fully Connected layers**. The output of CNN layers area concatenated and passed to the fully connected layers with sigmoid activation function.

## Model parameters
For convenience all parameters are stored in the config/config.yaml file. This file can be used to safely run previously trained model with all the prepocessing parameters.

## Text preprocessing
Each text runs through three stages: TextTokenizer, Encoder, and PadSequencer. 
TextTokenizer is tested with English, and Russian languages. It has multiple checks for the invalid simbols, words that sticked together, and many other cases.

## Examples
Please review the example notebook for reference. The example data used is taken from https://www.kaggle.com/c/quora-question-pairs.

