# snli_tensorflow
Various Tensorflow RNN models for Stanford Natural Language Inference (SNLI) Corpus


# How to download data

* Download [snli dataset](http://nlp.stanford.edu/projects/snli/).


# How to preprocess

`python3 preprocess.py`


# How to Train

`python3 main.py --mode=train --noload`

# How to Test


load 6000's step saved weights to run test. 

`python3 main.py --load_step=6000`

# Result

Reproducing (Bowman 15) result
> 100d LSTM RNN
> train 84.8, dev 77.6
>
> 150d LSTM RNN
> 


Reproducing (Rocktaschel 16) result
> Conditional encoding, shared: Train, Dev, Test : 72

Reproducing LSTMN (Cheng 16) result
> ????



# Thanks to

* Keras SNLI baseline example https://github.com/Smerity/keras_snli

* Attention model for entailment on SNLI corpus implemented in Tensorflow https://github.com/shyamupa/snli-entailment

* Tensorflow implementation of Long-Short-Term-Memory Network for Natural Language Inference https://github.com/vsitzmann/snli-attention-tensorflow

* Code template from here. https://github.com/allenai/bi-att-flow
