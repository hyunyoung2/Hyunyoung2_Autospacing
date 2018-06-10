"""This code is for example code about A single RNN(LSTM) Practice by Hyunyoung2

The code is made to understand the rnn execution on tensorflow by me.

Also this is Basic version for Autospacing about Korean Language.

Be careful about this information about the following code,

This code only run on 1 batch, So you would have to deal with

one sentence by one sentence to automatically space a sentence.
"""
### Version Check For tensorflow ###
import sys
import tensorflow as tf

# for generating a batch
import numpy as np
import random 

# For data setting for tensorflow graph ##
from collections import Counter

print("Python version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))
print("Numpy version: {}".format(np.__version__))

### Data setting for Tensorflow learning, evaluating, and predicting ###

# For embedding and predicting 
total_voca = None
total_label = None

# For training 
training_data = None
training_label = None

# For evaluating
evaluating_data = None
evaluating_label = None

# For predicting
predicting_data = None
predicting_label = None

# The original data

data = ["학교에갔었다", 
       "서울에도착",
       "학교에",
       "서울"]

ground_truths = ["BIIBII",
                "BIIBI",
                "BII",
                "BI"]

total_voca = training_data = evaluating_data = predicting_data = data
total_label = training_label = evaluating_label = predicting_label = ground_truths

## The total voca and label ###
print("===== The Original Data =======")
print("The total vocabularies: {}".format(total_voca))
print("The total labels: {}".format(total_label))

## The original data ###
print("===== The Original Data =======")
print("Data: {}".format(data))
print("Ground Truths: {}".format(ground_truths))

## The training data ###
print("===== The Training Data =======")
print("Data: {}".format(training_data))
print("Ground Truths: {}".format(training_label))

## The evaluating data ###
print("===== The Evaluating Data =======")
print("Data: {}".format(evaluating_data))
print("Ground Truths: {}".format(evaluating_label))

## The predicting data ###
print("===== The predicting Data =======")
print("Data: {}".format(predicting_data))
print("Ground Truths: {}".format(predicting_label))


### Generating Dictionary ####
def generate_feature_dict(total):
    print(type(total))
    # for Counter, data is join
    data_joined = "".join(total)
    print("Example of data_joined: {}, type: {}".format(data_joined,
                                                        type(data_joined)))
    #del data
    
    data_counter = Counter(data_joined)
    # removal of the duplication of syllable
    data_counter = Counter(data_joined)
    print("Example of data_counter: {}, type: {}, len: {}".format(data_counter,
                                                                   type(data_counter),
                                                                   len(data_counter)))
    del data_joined
    # make id per data, for example, data could be word and syllable
    # the index of a list is 
    data_list = [val for key, val in enumerate(data_counter)]
    # if you want to data list for idx per data, you don't need to use idx2data dict.
    # just use like data_list[data2idx("data")]
    # But here I used two the dictionaries for indexing for data, idx number.
    data2idx = {val:idx for idx, val in enumerate(data_list)}
    idx2data = {idx:val for idx, val in enumerate(data_list)}
    print("The list of data: {}, len: {}, type: {}".format(data_list,
                                                           len(data_list),
                                                           type(data_list)))
    print("data2idx: {}, len: {}, type: {}".format(data2idx, 
                                                   len(data2idx),
                                                   type(data2idx)))
    print("idx2data: {}, len: {}, type: {}".format(idx2data,
                                                   len(idx2data),
                                                   type(idx2data)))
    del data_list
    
    return data2idx, idx2data

# for data 
print("===== Data =======")
data2idx, idx2data = generate_feature_dict(total_voca)

# for label
print("\n\n===== Label =======")
label2idx, idx2label = generate_feature_dict(total_label)



### Make training data for tensorflow ####
def preprecossing_raw_data(data, dict_for_data):
    #print("data: {}".format(data))
    #print("dict_for_data: {}".format(dict_for_data))
    
    data_returned = list()
    
    # data is 2-dimensional for this function
    for idx, val in enumerate(data):
        # indexing_data is 1-dimensional
        indexing_data = list()
        for idx2, val2 in enumerate(val):
            indexing_data.append(dict_for_data[val2])
        
        data_returned.append(indexing_data)
    
    return data_returned

def change_label(label):
    
    label_changed = list()
    
    for idx, val in enumerate(label):
        temp_label = list()
        for idx2, val2 in enumerate(val):
            if val2 == 1:  # B tag
                temp_label.append([1, 0])
            else: # I tag
                temp_label.append([0, 1])
        label_changed.append(temp_label)
    
    return label_changed
        

# for training 
x_train = preprecossing_raw_data(training_data, data2idx)
y_train = preprecossing_raw_data(training_label, label2idx)
#y_train = change_label(y_train_before)

# for evaluating
x_evaluating = preprecossing_raw_data(evaluating_data, data2idx)
y_evaluating = preprecossing_raw_data(evaluating_label, label2idx)
#y_evaluating = change_label(y_evaluating_before)

# for predicting
x_predicting = preprecossing_raw_data(predicting_data, data2idx)
y_predicting = preprecossing_raw_data(predicting_label, label2idx)
#y_predicting = change_label(y_predicting_before)

## The training data ###
print("===== The Training Data =======")
print("Data: {}".format(x_train))
#print("Ground Truths: {}".format(y_train_before))
print("change_label: {}".format(y_train))

## The evaluating data ###
print("===== The Evaluating Data =======")
print("Data: {}".format(x_evaluating))
#print("Ground Truths: {}".format(y_evaluating_before))
print("change_label: {}".format(y_evaluating))

## The predicting data ###
print("===== The predicting Data =======")
print("Data: {}".format(x_predicting))
#print("Ground Truths: {}".format(y_predicting_before))
print("change_label: {}".format(y_predicting))


### Generate a batch ####

# shuffle function randomly
# if shuffle is true, shuffle 
# if not, don't shuffle
def shuffle_data(x_data, y_label, shuffle=False, debugging=False):
    x_data_shuffled = list()
    y_label_shuffled = list()
    
    data_zip = list(zip(x_data, y_label))
    
    if shuffle == True:
        random.shuffle(data_zip)
        
    for idx, val in enumerate(data_zip):
        x_data_shuffled.append(val[0])
        y_label_shuffled.append(val[1])
    
    if debugging: 
        print("========== suffling ============")
        print("x: {}".format(x_data_shuffled))
        print("y: {}".format(y_label_shuffled))
    
    return x_data_shuffled, y_label_shuffled

def generate_a_batch(x_data, y_label, batch_size=1, debugging=False):
    assert len(x_data) % batch_size == 0 
    
    batch_x_data = list()
    batch_y_label = list()
    
    for idx in range(len(x_data)//batch_size):
        batch_x_data.append(x_data[idx:idx+batch_size])
        batch_y_label.append(y_label[idx:idx+batch_size])
    
    if debugging:
        print("========== Batching :{} ============".format(batch_size))
        print("x: {}".format(batch_x_data))
        print("y: {}".format(batch_y_label))
    
    return batch_x_data, batch_y_label

## The training data ###
print("===== The Training Data =======")
shuffle_x_training, shuffl_y_training = shuffle_data(x_train, y_train, False, True)
batch_x_training1, batch_y_training1 = generate_a_batch(shuffle_x_training, shuffl_y_training, 1, True)
batch_x_training2, batch_y_training2 = generate_a_batch(shuffle_x_training, shuffl_y_training, 2, True)
## The evaluating data ###
print("===== The Evaluating Data =======")
shuffle_x_evaluating, shuffl_y_evaluating = shuffle_data(x_evaluating, y_evaluating, False, True)
batch_x_evaluating1, batch_y_evaluating1 = generate_a_batch(shuffle_x_evaluating, shuffl_y_evaluating, 1, True)
batch_x_evaluating2, batch_y_evaluating2 = generate_a_batch(shuffle_x_evaluating, shuffl_y_evaluating, 2, True)

## The predicting data ###
print("===== The predicting Data =======")
shuffle_x_predicting, shuffl_y_predicting = shuffle_data(x_predicting, y_predicting, False, True)
batch_x_evaluating1, batch_y_evaluating1 = generate_a_batch(shuffle_x_evaluating, shuffl_y_evaluating, 1, True)
batch_x_predicting2, batch_y_predicting2 = generate_a_batch(shuffle_x_predicting, shuffl_y_predicting, 2, True)

### Build Grapah for Tensorflow ###

############################
# Graph's Hyper parameters #
############################

# For word embedding
vocabulary_size = len(data2idx) # here, the number of syllable
print("Vacabulary size: {}".format(vocabulary_size))
embedding_size = 2

# a RNN and LSTM
batch_sizes = 1
time_steps = None
num_feature = embedding_size

# for Output Layer
num_classes = len(label2idx) # here, the number of tags, B and I, i.e.2

# for cell
num_hidden_units = num_classes

# for learning parameter
learning_rate = 0.1
epoch = 1
steps = 4 # total data size // batch size
log_location = "./log"

############################
#    Graph's input part    #
############################

new_batch_size = tf.placeholder(tf.int32, (), name="batch_size")

word_ids = tf.placeholder(tf.int32, (None, None), name="word_ids") # x data in (batch, time)
label_ids = tf.placeholder(tf.int32, (None, None), name="label_ids")

embedding_matrix = tf.get_variable("Embedding_matrix", shape=[vocabulary_size, embedding_size], dtype=tf.float32)
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids, name="word_embedding_lookup")

# for sparse label data
label_matrix = tf.constant([[1, 0],[0, 1]], dtype=tf.float32, name="label_matrix")
label_embeddings = tf.nn.embedding_lookup(label_matrix, label_ids, name="label_embedding_lookup")


x_inputs = word_embeddings # The original version = tf.placeholder(tf.float32, (None, None, num_features), name="Input") # (batch, time, in)
# tf.float32 is for cross entropy cost function.
y_labels = label_embeddings # tf.placeholder(tf.float32, (batch_sizes , time_steps, num_classes), name="output_label") # (batch, time, output)

print("The shape of new_batch_size:{}".format(new_batch_size.shape))

print("The shape of label_matrix:{}".format(label_matrix.shape))
print("The shape of label_ids:{}".format(label_ids.shape))
print("The shape of label_embeddings:{}".format(label_embeddings.shape))

print("The shape of embedding_matrix:{}".format(embedding_matrix.shape))
print("The shape of word_ids:{}".format(word_ids.shape))
print("The shape of word_embeddings:{}".format(word_embeddings.shape))

print("The shape of x_inputs:{}".format(x_inputs.shape))
print("The shape of y_labels:{}".format(y_labels.shape))

############################
#        Graph's RNN       #
############################

# defined in https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
rnn_type = ["BasicRNNCell", "BasicLSTMCell"]

# If you want to use another RNN, just change the following rnn_type
selected_rnn_type = rnn_type[1]

def make_cell(selected, cell_units):
    
    cell = None # 
    
    if selected == rnn_type[0]: 
        # Aliases : tf.contrib.rnn.BasicRNNCell
        # Most basic RNN: output = new_state = act(W * input(X_vector) + U * state(hidden) + B) in call function
        cell = tf.nn.rnn_cell.BasicRNNCell(cell_units, name=rnn_type[0])
    
    elif selected == rnn_type[1]:
        # Aliases : tf.contrib.BasicLSTMCell 
        cell = tf.nn.rnn_cell.BasicLSTMCell(cell_units, name=rnn_type[1])
        
    print("Cell type: {}".format(selected))
    return cell

my_rnn_cell = make_cell(selected_rnn_type, num_hidden_units)

# You don't need to initialize the initail_state, if you want zero_state
# that is by default zero_state
if selected_rnn_type == rnn_type[1]:
    my_initial = my_rnn_cell.zero_state(new_batch_size, dtype=tf.float32)

else:
    my_initial = tf.zeros([new_batch_size, my_rnn_cell.state_size], dtype=tf.float32)
    
outputs, state = tf.nn.dynamic_rnn(my_rnn_cell, x_inputs,
                                   initial_state=my_initial,
                                   dtype=tf.float32)


print("The shape of outputs:{}".format(outputs.shape))
print("The state:{}".format(state))
      
############################
#  Graph's Classfication   #
############################

# in here, we don't need to classify them, just use probability distribution

#softmax_outputs_of_rnn = tf.nn.softmax(outputs)

predictions = outputs

print("The shape of predictions:{}".format(predictions.shape))

############################
#      Graph's Loss        #
############################

# https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2
loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y_labels)
loss_mean_op = tf.reduce_mean(loss_op)
tf.summary.scalar("Loss", loss_mean_op)

############################
#    Graph's optimizer     #
############################

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
traing_op = optimizer.minimize(loss_mean_op)


############################
#    Graph's Accuracy      #
############################

# predictions_prob_distributed means the probability to each label. 
predictions_prob_distributed = tf.nn.softmax(predictions)

correct_prediction = tf.equal(tf.argmax(predictions_prob_distributed, 2), tf.argmax(y_labels, 2))
true_or_not = tf.cast(correct_prediction, dtype=tf.float32)
accuracy = tf.reduce_mean(true_or_not)
accuracy_per_line = tf.reduce_sum(true_or_not)
tf.summary.scalar("Accuracy", accuracy)

############################
#    Graph's Prediction    #
############################

# Keep in mind that thit prediction is for one sentence by one sentence

# the original tensor is a tensor of shape (batch, time step, output class)
# the tensor reshapes the original one to (batch, output class)
pro_reshaped = tf.reshape(predictions_prob_distributed, [-1, 2], name="To_predict")

tag_predicted = tf.argmax(pro_reshaped, 1)

######################################
#    Graph's initializing variable   #
######################################

init_op = tf.global_variables_initializer()

######################################
# Graph's summary merging operation  #
######################################

merged_op = tf.summary.merge_all()

###################################
#    Graph's Checking variable    #
###################################
trainable_variable1 = tf.get_collection("tf.GraphKeys.GLOBAL_VARIABLES")
trainable_variable2 = tf.get_collection("tf.GraphKeys.TRAINABLE_VARIABLES")
trainable_variable3 = tf.get_collection("tf.GraphKeys.LOCAL_VARIABLES")
sess = tf.Session()
sess.run(init_op)
print("===== variable type =====")
print("tf.GraphKeys.GLOBAL_VARIABLES: {}".format(sess.run([trainable_variable1,trainable_variable1])))
print("tf.GraphKeys.TRAINABLE_VARIABLES: {}".format(sess.run([trainable_variable1,trainable_variable2])))
print("tf.GraphKeys.LOCAL_VARIABLES: {}".format(sess.run([trainable_variable1,trainable_variable3])))

print("\n\n===== all variables =====")
for v in tf.global_variables():
    print(v.name)


#### learning Tenssorflow  ####

with tf.Session() as sess:
    sess.run(init_op)
    
    train_writer = tf.summary.FileWriter(log_location+"/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_location+"/test")
    prediction_writer = tf.summary.FileWriter(log_location+"/prediction")
    
    # Let's training
    print("================Let's train!!==============")
    for current_epoch in range(epoch):
        shuffle_x_training, shuffl_y_training = shuffle_data(x_train, y_train, 
                                                             False, True)
        batch_x_training, batch_y_training = generate_a_batch(shuffle_x_training, 
                                                               shuffl_y_training, 
                                                               batch_sizes, True)
        print("The epoch: {}".format(current_epoch))
        print("The number of batchs: {}".format(len(batch_x_training)))
        for idx in range(len(batch_x_training)):
            print("====================== my batch data ============================")
            print("idx: {}, x: {}, y: {}".format(idx, batch_x_training[idx], batch_y_training[idx]))
            summary, embedding_matrix_, word_embeddings_, predictions_, prob_ , label_matrix_, loss_mean_op_, _ = sess.run([merged_op, embedding_matrix, word_embeddings, predictions, predictions_prob_distributed, label_matrix, loss_mean_op, traing_op], 
                                                                                                                  feed_dict={new_batch_size: batch_sizes, word_ids: batch_x_training[idx], label_ids: batch_y_training[idx]})
            print("loss_mean_op:{}".format(loss_mean_op_))
            print("label_matrix: {}".format(label_matrix_))
            print("predictions_:{}".format(predictions_))
            print("prob_:{}".format(prob_))
            print("embedding_matrix:{}".format(embedding_matrix_))
            print("word_embeddings: {}".format(word_embeddings_))
            train_writer.add_summary(summary, idx)
            
    # Let's evaluting
    print("\n\n================ Let's evaluate!!==============")
    for current_epoch in range(epoch):
        shuffle_x_evaluating, shuffl_y_evaluating = shuffle_data(x_evaluating, y_evaluating, 
                                                             False, True)
        batch_x_evaluating, batch_y_evaluating = generate_a_batch(shuffle_x_evaluating, 
                                                               shuffl_y_evaluating, 
                                                               batch_sizes, True)
        print("The epoch: {}".format(current_epoch))
        print("The number of batchs: {}".format(len(batch_x_evaluating)))
        total_accuracy = 0.0
        len_label = 0
        for idx in range(len(batch_x_evaluating)):
            print("====================== my batch data ============================")
            print("idx: {}, x: {}, y: {}".format(idx, batch_x_evaluating[idx], batch_y_evaluating[idx]))
            len_label += len(batch_y_evaluating[idx][0])
            
            summary, accuracy_, accuracy_per_line_ = sess.run([merged_op, accuracy, accuracy_per_line], 
                                                     feed_dict={new_batch_size: batch_sizes, 
                                                                word_ids: batch_x_evaluating[idx], 
                                                                label_ids: batch_y_evaluating[idx]})
            
            print("Accuracy for each sentence in data: {}".format(accuracy_))
            total_accuracy += accuracy_per_line_
            test_writer.add_summary(summary, idx)
            
        print("The current epoch({}) accuracy: {}".format(current_epoch, (total_accuracy/len_label)))
        
    # Let's predict
    print("\n\n================ Let's predict!!==============")
    for current_epoch in range(epoch):
        shuffle_x_predicting, shuffl_y_predicting = shuffle_data(x_predicting, y_predicting, 
                                                             False, True)
        batch_x_predicting, batch_y_predicting = generate_a_batch(shuffle_x_predicting, 
                                                               shuffl_y_predicting, 
                                                               batch_sizes, True)
        print("The epoch: {}".format(current_epoch))
        print("The number of batchs: {}".format(len(batch_x_predicting)))
        len_label = 0
        for idx in range(len(batch_x_predicting)):
            print("====================== my batch data ============================")
            print("idx: {}, x: {}, y: {}".format(idx, batch_x_predicting[idx], batch_y_predicting[idx]))
            
            summary, pro_reshaped_, tag_predicted_ = sess.run([merged_op, pro_reshaped, tag_predicted],
                                                     feed_dict={new_batch_size: batch_sizes, 
                                                                word_ids: batch_x_predicting[idx], 
                                                                label_ids: batch_y_evaluating[idx]})
            prediction_writer.add_summary(summary, idx)
        
            print("prediction:{}".format(tag_predicted_))
            
    train_writer.close()
    test_writer.close()
    prediction_writer.close()
