from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, TFBertModel

import unicodedata
import re
import numpy as np
import os
import io
import time
import datetime

import json



train_flag = 1
path_to_file = "data/squad_ent_short.json"
num_examples = 10000
BATCH_SIZE = 64
units = 100 #512
EPOCHS = 4

bert_model = TFBertModel.from_pretrained('bert-base-uncased')
vocab_size=30522
embedding_dim=768



def max_length(tensor):
  return max(len(t) for t in tensor)





# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
#train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
#test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
#### tensorboard ####



def tensor(input, tokenizer):
    tensor=tokenizer.encode(input)
    # for i in range(len(tensor)):
    #     a=[]
    #     tensor[i]=[tensor[i]]
    #tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
    return tensor

def load_data(path, num_examples, tokenizer):
    data=""
    with open(path, 'r') as file:
        json_file = json.load(file)
    data = json_file.get('data')

    input_tensor=[]
    target_tensor=[]
    answer_tensor=[]
    i=0

    for qas in data:
        sentence= qas.get("sentence")
        input_tensor.append(tensor(sentence,tokenizer))

        answer = qas.get("answer")
        answer_tensor.append(tensor(answer,tokenizer))

        question=qas.get("question")
        target_tensor.append(tensor(question,tokenizer))

        i+=1
        if i>=num_examples:
            break


    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,padding='post')
    answer_tensor = tf.keras.preprocessing.sequence.pad_sequences(answer_tensor,padding='post')

    return input_tensor,target_tensor, answer_tensor


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_tensor,target_tensor, answer_tensor = load_data(path_to_file, num_examples,tokenizer)



# print(input_tensor)
#
# input= input_tensor[0]
# print(input)
#
# print(len(input))
# print(len(input_tensor[1]))
# print(len(input_tensor[2]))
# print(len(input_tensor[3]))
# for token in input:
#     print(tokenizer.decode(token)+" ")
#
# print(maxlen)


# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val, answer_tensor_train, answer_tensor_val = train_test_split(input_tensor, target_tensor,answer_tensor, test_size=0.2)

# Show length
#print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))



def convert(tokenizer, tensor):
   for t in tensor:
     if t!=0:
       print ("%d ----> %s" % (t, tokenizer.encode(t)))

'''
# print ("Input Context; index to word mapping")
# convert(tokenizer, input_tensor_train[0])
# print ()
# print ("Target Question; index to word mapping")
# convert(tokenizer, target_tensor_train[0])
'''

#print (len(tokenizer.word_index)+1)


BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE #modify for performance

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train, answer_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

dataset_test = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val, answer_tensor_val))
dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)

#example_input_batch, example_target_batch = next(iter(dataset))

#print(example_input_batch.shape)
#print("\n")
#print(example_target_batch.shape)

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = bert_model
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    x= x[0]
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)

# sample input
#sample_hidden = encoder.initialize_hidden_state()
#sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
#print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
#print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

#attention_layer = BahdanauAttention(10)
#attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = bert_model
        self.gru = tf.keras.layers.GRU(self.dec_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, encoded_answer):

        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        #not using the pool embedding
        x=x[0]
        encoded_answer = tf.cast(encoded_answer,tf.float32)
        x = tf.concat([tf.expand_dims(context_vector, 1), x, tf.expand_dims(encoded_answer, 1)], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights

decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)



@tf.function
def train_step(inp, targ, ans, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    out,encoded_answer=encoder(ans, enc_hidden)
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    #print(tokenizer.encode("[CLS]"))
    dec_input = tf.expand_dims([0] * BATCH_SIZE,1)
    #dec_input = tf.expand_dims(BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, encoded_answer)
      loss += loss_function(targ[:, t], predictions)
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  #tensorboard
  train_loss(loss)
  #test_accuracy(inp, targ)

  return batch_loss

if train_flag == 1:

    for epoch in range(EPOCHS):
      start = time.time()

      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0

    #train
      for (batch, (inp, targ, ans)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, ans, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy()))




      # saving (checkpoint) the model every epoch

      checkpoint.save(file_prefix = checkpoint_prefix)

      print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    #reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    #train_accuracy.reset_states()
    #test_accuracy.reset_states()
