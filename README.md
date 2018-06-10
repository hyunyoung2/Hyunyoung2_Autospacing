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
 
   1. [A Single RNN or LSTM](https://nbviewer.jupyter.org/github/hyunyoung2/Hyunyoung2_Autospacing/blob/master/A_Single_RNN_or_LSTM/A_Single_RNN_Or_LSTM.ipynb) 

 
 - Python code version 
 
   1. [A Single RNN or LSTM](https://github.com/hyunyoung2/Hyunyoung2_Autospacing/blob/master/A_Single_RNN_or_LSTM/A_single_RNN_Or_LSTM.py))



 # Reference paper and site
 
  1. [Wikipedia abut BIO tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
 
  2. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991v1)
