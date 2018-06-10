# Korean Language Autospacing

 This repository is for research about Korean Language word segmentation(autospacing).
 
 The following is an simple example about data for autospacing. 

 | 나 | 는 |   | 밥 | 을 |   | 먹 | 고 |   | 학 | 교 | 에 |   | 갔 | 다 | . |
 |----|----|---|----|----|---|----|----|---|----|----|----|---|----|----|---|
 | B  | I  |   | B  | I  |   | B  | I  |   | B  | I  | I  |   | B  | I  | I |

 here code is for practice of RNN(LSTM) on Tensorflow. 
 
 So, the way to learn RNN is based on stochastic gradient descent. And This code only use a batch. 

 ### The Example code 

 - Jupyter notebook version 
 
   1. [A Single RNN or LSTM](https://nbviewer.jupyter.org/github/hyunyoung2/Hyunyoung2_Autospacing/blob/master/A_Single_RNN_or_LSTM/A_Single_RNN_Or_LSTM.ipynb) under the A_Single_RNN_or_LSTM directory 

   2. [Multiple layers RNN or LSTM](https://nbviewer.jupyter.org/github/hyunyoung2/Hyunyoung2_Autospacing/blob/master/Multiple_layers_RNN_or_LSTM/Multiple_layers_RNN_or_LSTM.ipynb) under the Multiple_layers_RNN_or_LSTM directory

   3. [Single Bi-RNN or Bi-LSTM](https://nbviewer.jupyter.org/github/hyunyoung2/Hyunyoung2_Autospacing/blob/master/Multiple_layers_RNN_or_LSTM/Multiple_layers_RNN_or_LSTM.ipynb) under the Single_Bi_RNN_or_LSTM directory

 - Python code version 
 
   1. [A Single RNN or LSTM](https://github.com/hyunyoung2/Hyunyoung2_Autospacing/blob/master/A_Single_RNN_or_LSTM/A_Single_RNN_Or_LSTM.py) under the A_Single_RNN_or_LSTM directory

   2. [Multiple layers RNN or LSTM](https://github.com/hyunyoung2/Hyunyoung2_Autospacing/blob/master/Multiple_layers_RNN_or_LSTM/Multiple_layers_RNN_or_LSTM.py) under the Multiple_layers_RNN_or_LSTM directory
   
   3. [Single Bi-RNN or Bi-LSTM](https://github.com/hyunyoung2/Hyunyoung2_Autospacing/blob/master/Single_Bi_RNN_or_LSTM/A_Single_Bi_RNN_or_Bi_LSTM.py) under the Single_Bi_RNN_or_LSTM directory

 # Reference paper and site
 
  1. [Understanding LSTM of colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  
  2. [Wikipedia abut BIO tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
 
  3. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991v1)

  4. [Recurrent neural networks and LSTM tutorial in Python and TensorFlow](http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/)

  5. [RNN Cell types of Tensorflow APIs](https://www.tensorflow.org/versions/r1.8/api_guides/python/contrib.rnn#Core_RNN_Cells_for_use_with_TensorFlow_s_core_RNN_methods)
