import tensorflow as tf
import numpy as np
from .utils.layer_utils import *
from .utils.match_utils import *
from .utils.data_utils import Batch 

class BiMPM():
    def __init__(self, sess, args, vocab_char, vocab_word):
        self.sess = sess
        self.dim_word = args.dim_word
        self.dim_char_embed = args.dim_char_embed
        self.dim_char_lstm = args.dim_char_lstm
        self.num_chars = args.num_chars
        self.dim_input = self.dim_word + self.dim_char_lstm * 2
        self.aggregation_lstm_dim = 100
        self.num_labels = args.num_labels
        self.lr = args.lr
        self.dropout = args.dropout
        self.task = args.task
        self.use_sgd = args.use_sgd

        vocab_char.build_embed("char_embedding")
        vocab_word.build_embed("word_embedding", trainable=False)        

        self._build_input()
        self._build_embed()
        with tf.variable_scope("m_net", initializer=tf.random_uniform_initializer(-0.01, 0.01)):
            self._build_char_representation_layers()
            self._build_word_representation_layers()
            self._build_matching_layers()

    def _build_input(self):
        self.in_a_lengths = tf.placeholder(tf.int32, [None])
        self.in_b_lengths = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.in_a_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, a_len]
        self.in_b_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, b_len]

        self.a_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, a_len]
        self.b_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, b_len]
        self.in_a_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, a_len, q_char_len]
        self.in_b_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, b_len, p_char_len]

        self.dropout_rate = tf.placeholder_with_default(0.0, ())

        self.batch_size = tf.shape(self.label)[0]
        self.a_len = tf.shape(self.in_a_words)[1]
        self.b_len = tf.shape(self.in_b_words)[1]  

        self.in_a_rep, self.in_b_rep = [], []     

        self.a_lengths = self.in_a_lengths
        self.b_lengths = self.in_b_lengths

    def _build_embed(self):
        with tf.variable_scope("embedding", reuse=True):
            self.embed_char = tf.get_variable("char_embedding")       
            self.embed_word = tf.get_variable("word_embedding")

    def _build_char_representation_layers(self):
        a_char_len = tf.shape(self.in_a_chars)[2]
        b_char_len = tf.shape(self.in_b_chars)[2]

        self.in_a_char_rep = tf.nn.embedding_lookup(self.embed_char, self.in_a_chars) # [batch_size, a_len, a_char_len, dim_char]
        self.in_a_char_rep = tf.reshape(self.in_a_char_rep, shape=[-1, a_char_len, self.dim_char_embed])
        a_char_lengths = tf.reshape(self.a_char_lengths, [-1])
        a_char_mask = tf.sequence_mask(a_char_lengths, a_char_len, dtype=tf.float32)  # [batch_size*a_len, a_char_len]
        self.in_a_char_rep = tf.multiply(self.in_a_char_rep, tf.expand_dims(a_char_mask, axis=-1))

        self.in_b_char_rep = tf.nn.embedding_lookup(self.embed_char, self.in_b_chars) # [batch_size, b_len, b_char_len, dim_char]
        self.in_b_char_rep = tf.reshape(self.in_b_char_rep, shape=[-1, b_char_len, self.dim_char_embed])
        b_char_lengths = tf.reshape(self.b_char_lengths, [-1])
        b_char_mask = tf.sequence_mask(b_char_lengths, b_char_len, dtype=tf.float32)  # [batch_size*b_len, b_char_len]
        self.in_b_char_rep = tf.multiply(self.in_b_char_rep, tf.expand_dims(b_char_mask, axis=-1))

        (a_char_outputs_fw, a_char_outputs_bw, _) = my_lstm_layer(self.in_a_char_rep, self.dim_char_lstm,
                input_lengths=a_char_lengths,scope_name="char_lstm", reuse=False, dropout_rate=self.dropout_rate)
        a_char_outputs_fw = collect_final_step_of_lstm(a_char_outputs_fw, a_char_lengths - 1)
        a_char_outputs_bw = a_char_outputs_bw[:, 0, :]
        a_char_outputs = tf.concat(axis=1, values=[a_char_outputs_fw, a_char_outputs_bw])
        a_char_outputs = tf.reshape(a_char_outputs, [self.batch_size, self.a_len, 2*self.dim_char_lstm])

        (b_char_outputs_fw, b_char_outputs_bw, _) = my_lstm_layer(self.in_b_char_rep, self.dim_char_lstm,
                input_lengths=b_char_lengths, scope_name="char_lstm", reuse=True, dropout_rate=self.dropout_rate)
        b_char_outputs_fw = collect_final_step_of_lstm(b_char_outputs_fw, b_char_lengths - 1)
        b_char_outputs_bw = b_char_outputs_bw[:, 0, :]
        b_char_outputs = tf.concat(axis=1, values=[b_char_outputs_fw, b_char_outputs_bw])
        b_char_outputs = tf.reshape(b_char_outputs, [self.batch_size, self.b_len, 2*self.dim_char_lstm])
            
        self.in_a_rep.append(a_char_outputs)
        self.in_b_rep.append(b_char_outputs)  

    def _build_word_representation_layers(self):
        self.in_a_word_rep = tf.nn.embedding_lookup(self.embed_word, self.in_a_words) # [batch_size, a_len, dim_word]
        self.in_b_word_rep = tf.nn.embedding_lookup(self.embed_word, self.in_b_words) # [batch_size, b_len, dim_word]

        self.in_a_rep.append(self.in_a_word_rep)
        self.in_b_rep.append(self.in_b_word_rep)

        self.in_a_rep = tf.concat(axis=2, values=self.in_a_rep) # [batch_size, a_len, dim]
        self.in_b_rep = tf.concat(axis=2, values=self.in_b_rep) # [batch_size, b_len, dim]

        self.in_a_rep = tf.nn.dropout(self.in_a_rep, (1 - self.dropout_rate))
        self.in_b_rep = tf.nn.dropout(self.in_b_rep, (1 - self.dropout_rate))

        self.mask = tf.sequence_mask(self.b_lengths, self.b_len, dtype=tf.float32) # [batch_size, b_len]
        self.a_mask = tf.sequence_mask(self.a_lengths, self.a_len, dtype=tf.float32) # [batch_size, a_len]

        with tf.variable_scope("input_highway"):
            self.in_a_rep = multi_highway_layer(self.in_a_rep, self.dim_input, 1)
            tf.get_variable_scope().reuse_variables()
            self.in_b_rep = multi_highway_layer(self.in_b_rep, self.dim_input, 1)

        self.in_a_rep = tf.multiply(self.in_a_rep, tf.expand_dims(self.a_mask, axis=-1))
        self.in_b_rep = tf.multiply(self.in_b_rep, tf.expand_dims(self.mask, axis=-1))

    def _build_matching_layers(self):
        # ========Bilateral Matching=====
        (match_representation, match_dim) = bilateral_match_func(self.in_a_rep, self.in_b_rep,
                        self.a_lengths, self.b_lengths, self.a_mask, self.mask, self.dim_input, dropout_rate=self.dropout_rate)

        # #========Prediction Layer=========
        match_dim = 4 * self.aggregation_lstm_dim
        w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim/2, self.num_labels],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [self.num_labels],dtype=tf.float32)

        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.tanh(logits)
        logits = tf.nn.dropout(logits, (1 - self.dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1

        self.logits = logits
        self.prob = tf.nn.softmax(logits)
        
        gold_matrix = tf.one_hot(self.label, self.num_labels, dtype=tf.float32)
        self.loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix)
        self.loss = tf.reduce_mean(self.loss_all)

        correct = tf.nn.in_top_k(logits, self.label, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.predictions = tf.cast(tf.argmax(self.prob, 1), tf.int32)
        self.score = 1 - self.prob[:, 0]
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.label), dtype=tf.float32))

        self.params = []
        print("Trainable variables of M_Net:")
        for var in tf.trainable_variables():
            if var.name.find("m_net/") == 0 or var.name.find("embedding/") == 0:
                self.params.append(var)
                print(var)
        print()

        if self.use_sgd:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads = compute_gradients(self.loss, self.params)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.train_op = optimizer.apply_gradients(zip(grads, self.params))

    def step(self, batch, is_train=False, attack=False):
        _batch = batch
        
        batch = Batch(batch)
        input_feed = {
            self.in_a_lengths: batch.a_lengths,
            self.in_b_lengths: batch.b_lengths,
            self.in_a_words: batch.in_a_words,
            self.in_b_words: batch.in_b_words,
            self.label: batch.label,
            self.a_char_lengths: batch.a_char_lengths,
            self.b_char_lengths: batch.b_char_lengths,
            self.in_a_chars: batch.in_a_chars,
            self.in_b_chars: batch.in_b_chars,
        }

        output_feed = [self.loss, self.accuracy, self.predictions, self.score]
        if is_train:
            input_feed[self.dropout_rate] = self.dropout
            output_feed.append(self.train_op)

        res = self.sess.run(output_feed, input_feed)

        return res