# A Single RNN or LSTM

As for this single RNN(LSTM), 

This code is for example code about A single RNN(LSTM) Practice by Hyunyoung2

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

> python3 A_Single_RNN_Or_LSTM.py

> The following is my computational graph of A singel RNN(LSTM) for Korean Autospacing. 

![](https://raw.githubusercontent.com/hyunyoung2/Hyunyoung2_Autospacing/master/A_Single_RNN_or_LSTM/A_single_RNN_graph_of_my_model.png)

## The example code

- [(Jupyter notebook version)](https://nbviewer.jupyter.org/github/hyunyoung2/Hyunyoung2_Autospacing/blob/master/A_Single_RNN_or_LSTM/A_Single_RNN_Or_LSTM.ipynb) 
- [(python code version)](https://github.com/hyunyoung2/Hyunyoung2_Autospacing/blob/master/A_Single_RNN_or_LSTM/A_Single_RNN_Or_LSTM.py)

# Reference
  
 - [For BasicLSTMCell, THe paper](https://arxiv.org/abs/1409.2329v5)

 - [Recurrent Neural Networks Tutrorial of Tensorflow](https://www.tensorflow.org/tutorials/recurrent)

 - Tensorflow API: 
  
   - [BasicRNNCell](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/BasicRNNCell)
  
   - [BasicLSTMCell](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/contrib/rnn/BasicLSTMCell)
  
   - [DynamicRNN](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/nn/dynamic_rnn)

 - blogs about RNN
 
   - [MONIK"S BLOG](http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/)
   
   - [Recurrent Neural Networks in Tensorflow II](https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html)
   
   - [Kor ver. usage of tensorflow API about RNN](https://kakalabblog.wordpress.com/2017/06/23/implementing-rnn-in-tensorflow/)
   
   - [LSTM by Example using Tensorflow](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537)
   
   - [Understanding LSTM in Tensorflow(MNIST dataset)](https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)
   
   - [How to build a Recurrent Neural Network in TensorFlow (1/7)](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)
   
 - Stackover flow questions about RNN: 
 
   - [The difference between dynamic and rnn](https://stackoverflow.com/questions/39734146/whats-the-difference-between-tensorflow-dynamic-rnn-and-rnn)
 
 - Another example codes
 
   - [aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)
