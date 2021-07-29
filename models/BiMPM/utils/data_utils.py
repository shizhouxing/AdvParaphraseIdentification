import numpy as np
import re

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
        out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val

def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in range(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in range(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val

class Batch:
    def __init__(self, batch, label=None):
        if label is not None:
            _batch = batch
            batch = []
            for sample in _batch:
                if sample["label"] == label:
                    batch.append(sample)
        self.batch_size = len(batch)
        
        self.a_len = 0
        self.b_len = 0
        self.a_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_a_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, a_len]
        self.b_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_b_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, b_len]
        self.label = []  # [batch_size]

        self.in_a_chars = [] # tf.placeholder(tf.int32, [None, None, None])  # [batch_size, a_len, q_char_len]
        self.a_char_lengths = [] # tf.placeholder(tf.int32, [None, None])  # [batch_size, a_len]
        self.in_b_chars = [] # tf.placeholder(tf.int32, [None, None, None])  # [batch_size, b_len, p_char_len]
        self.b_char_lengths = [] # tf.placeholder(tf.int32, [None, None])  # [batch_size, b_len]

        for sample in batch:
            cur_a_length = len(sample["word_idx_a"])
            cur_b_length = len(sample["word_idx_b"])
            if self.a_len < cur_a_length: self.a_len = cur_a_length
            if self.b_len < cur_b_length: self.b_len = cur_b_length
            self.a_lengths.append(cur_a_length)
            self.in_a_words.append(sample["word_idx_a"])
            self.b_lengths.append(cur_b_length)
            self.in_b_words.append(sample["word_idx_b"])
            self.label.append(sample["label"])
            self.in_a_chars.append(sample["char_idx_a"])
            self.in_b_chars.append(sample["char_idx_b"])
            self.a_char_lengths.append([len(cur_char_idx) for cur_char_idx in sample["char_idx_a"]])
            self.b_char_lengths.append([len(cur_char_idx) for cur_char_idx in sample["char_idx_b"]])

        # padding all value into np arrays
        self.a_lengths = np.array(self.a_lengths, dtype=np.int32)
        self.in_a_words = pad_2d_vals(self.in_a_words, self.batch_size, self.a_len, dtype=np.int32)
        self.b_lengths = np.array(self.b_lengths, dtype=np.int32)
        self.in_b_words = pad_2d_vals(self.in_b_words, self.batch_size, self.b_len, dtype=np.int32)
        self.label = np.array(self.label, dtype=np.int32)
        max_char_length1 = np.max([np.max(aa) for aa in self.a_char_lengths])
        self.in_a_chars = pad_3d_vals(self.in_a_chars, self.batch_size,  self.a_len,
                                                max_char_length1, dtype=np.int32)
        max_char_length2 = np.max([np.max(aa) for aa in self.b_char_lengths])
        self.in_b_chars = pad_3d_vals(self.in_b_chars, self.batch_size,  self.b_len,
                                            max_char_length2, dtype=np.int32)

        self.a_char_lengths = pad_2d_vals(self.a_char_lengths, self.batch_size,  self.a_len)
        self.b_char_lengths = pad_2d_vals(self.b_char_lengths, self.batch_size,  self.b_len)