# Multiple layers RNN or LSTM

As for this Multiple layers RNN(LSTM),

This code is for example code about multiple layers RNN(LSTM) Practice by Hyunyoung2

But, here This code would use 2 layers. 

The code is made to understand the rnn execution on tensorflow by me.

Also this is Basic version for Autospacing about Korean Language.

Be careful about this information about the following code,

This code only run on 1 batch, So you would have to deal with

one sentence by one sentence to automatically space a sentence.

We would show you two version for RNN(LSTM). 

The rnn_type variable could select the version of RNN as following:

```
# defined in https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
rnn_type = ["BasicRNNCell", "BasicLSTMCell"]

# If you want to use another RNN, just change the following rnn_type
selected_rnn_type = rnn_type[1]
```

If you want to run the code I made, type in like :

> python3 Multiple_layers_RNN_or_LSTM.py

> The following is my computational graph of A singel RNN(LSTM) for Korean Autospacing. 

![](https://raw.githubusercontent.com/hyunyoung2/Hyunyoung2_Autospacing/master/Multiple_layers_RNN_or_LSTM/Multiple_layers_RNN_or_LSTM.png)

## The example code

- [(Jupyter notebook version)](https://nbviewer.jupyter.org/github/hyunyoung2/Hyunyoung2_Autospacing/blob/master/Multiple_layers_RNN_or_LSTM/Multiple_layers_RNN_or_LSTM.ipynb) 

- [(python code version)](https://github.com/hyunyoung2/Hyunyoung2_Autospacing/blob/master/Multiple_layers_RNN_or_LSTM/Multiple_layers_RNN_or_LSTM.py)

# Reference
  
 - [For BasicLSTMCell, THe paper](https://arxiv.org/abs/1409.2329v5)

 - [Recurrent Neural Networks Tutrorial of Tensorflow](https://www.tensorflow.org/tutorials/recurrent)

 - Tensorflow API: 
  
   - [BasicRNNCell](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/BasicRNNCell)
  
   - [BasicLSTMCell](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/BasicLSTMCell)
   
   - [MultipleRNNCell](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/MultiRNNCell)
  
   - [DynamicRNN](https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/dynamic_rnn)
