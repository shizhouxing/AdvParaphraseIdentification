import tensorflow as tf
import os, importlib, random, gzip, pickle, nltk, time, copy
from .util import logger
from .util.parameters import load_parameters
from .util.data_processing import *
from .util.evaluate import *
from .models.DIIN import MyModel as MyModel
from utils import load_data_raw
import numpy as np
from multiprocessing import Pool

def get_ner(example):
    def get(pos):
        chunk = ne_chunk(pos)
        ner = []
        for i in range(len(chunk)):
            if type(chunk[i]) == nltk.tree.Tree:
                if chunk[i].label() in ner_dict:
                    ner.append([i, ner_dict[chunk[i].label()]])
        return ner
    example["sentence1_NER_feature"] = get(example["sent_a_pos"])
    example["sentence2_NER_feature"] = get(example["sent_b_pos"])
    return example

class DIIN:
    def __init__(self, sess, args):
        self.FIXED_PARAMETERS, self.config = load_parameters()
        self.load_data(args)
        self.sess = sess
        self.learning_rate =  self.FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step = self.config.display_step
        self.eval_step = self.config.eval_step
        self.save_step = self.config.eval_step
        self.embedding_dim = self.FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = self.FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = self.FIXED_PARAMETERS["batch_size"]
        self.emb_train = self.FIXED_PARAMETERS["emb_train"]
        self.keep_rate = self.FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = self.FIXED_PARAMETERS["seq_length"] 
        self.model = MyModel(self.config, seq_length=self.sequence_length, emb_dim=self.embedding_dim,  hidden_dim=self.dim, embeddings=self.loaded_embeddings, emb_train=self.emb_train)
        self.cpus = args.cpus
        self.em_cache = {}

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.model.total_cost, tvars), config.gradient_clip_value)
        opt = tf.train.AdadeltaOptimizer(args.lr)
        self.optimizer = opt.apply_gradients(zip(grads, tvars), global_step=self.model.global_step)

        self.loss = self.model.total_cost
        self.accuracy = self.model.acc
        self.predictions = self.model.predictions
        self.error = self.model.error

    def load_data(self, args):
        path = "cache/data_%s_diin_private.pkl.gz" % args.data
        if os.path.exists(path):
            print("Loading cached dataset from %s" % path)
            f = gzip.open(path, "rb")
            self.data_train, self.data_valid, self.data_test, \
            self.indices_to_words, self.word_indices, \
            self.char_indices, self.indices_to_chars = pickle.load(f)
            f.close()
        else:
            self.data_train = load_data_raw(args, "train")
            self.data_valid = load_data_raw(args, "dev")
            self.data_test = load_data_raw(args, "test")
            datasets = [self.data_train, self.data_valid, self.data_test]

            print("Processing data for DIIN...")
            for dataset in datasets:
                for i, sample in enumerate(dataset):
                    sample["sent_a"] = sample["sent_a"][:self.config.seq_length]
                    sample["sent_b"] = sample["sent_b"][:self.config.seq_length]
                    sample["sent_a_pos"] = sample["sent_a_pos"][:self.config.seq_length]
                    sample["sent_b_pos"] = sample["sent_b_pos"][:self.config.seq_length]
                    sample["sentence1_binary_parse"] = " ".join(sample["sent_a"])
                    sample["sentence2_binary_parse"] = " ".join(sample["sent_b"])
                    sample["pairID"] = i

            datasets, self.indices_to_words, self.word_indices, self.char_indices, \
                self.indices_to_chars = sentences_to_padded_index_sequences(datasets)
            self.config.char_vocab_size = len(self.char_indices.keys())        
            self.data_train, self.data_valid, self.data_test = datasets
            
            f = gzip.open(path, 'wb')
            pickle.dump((
                self.data_train, self.data_valid, self.data_test,
                self.indices_to_words, self.word_indices,
                self.char_indices, self.indices_to_chars
            ), f)
            f.close()            

        print("Loading embeddings...")
        embedding_path = "cache/vocab_%s_diin_private.pkl.gz" % args.data

        if os.path.exists(embedding_path):
            f = gzip.open(embedding_path, 'rb')
            self.loaded_embeddings = pickle.load(f)
            f.close()
        else:
            glove_path = args.word_vector
            self.loaded_embeddings = loadEmbedding_rand(glove_path, self.word_indices)
            f = gzip.open(embedding_path, 'wb')
            pickle.dump(self.loaded_embeddings, f)
            f.close()

        self.config.char_vocab_size = len(self.char_indices.keys())            

    def get_minibatch(self, batch, training=False):
        indices = range(len(batch))

        genres = [['quora'] for i in indices]
        labels = [batch[i]['label'] for i in indices]

        if not "pairID" in batch[0]:
            pairIDs = np.array(range(len(batch)))
        else:
            pairIDs = np.array([batch[i]['pairID'] for i in indices])
            
        premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] * len(indices)

        premise_vectors = fill_feature_vector_with_cropping_or_padding([batch[i]['sentence1_binary_parse_index_sequence'][:] for i in indices], premise_pad_crop_pair, 1)
        hypothesis_vectors = fill_feature_vector_with_cropping_or_padding([batch[i]['sentence2_binary_parse_index_sequence'][:] for i in indices], hypothesis_pad_crop_pair, 1)

        premise_pos_vectors = generate_quora_pos_feature_tensor([batch[i]['sentence1_part_of_speech_tagging'][:] for i in indices], premise_pad_crop_pair)
        hypothesis_pos_vectors = generate_quora_pos_feature_tensor([batch[i]['sentence2_part_of_speech_tagging'][:] for i in indices], hypothesis_pad_crop_pair)

        premise_char_vectors = fill_feature_vector_with_cropping_or_padding([batch[i]['sentence1_binary_parse_char_index'][:] for i in indices], premise_pad_crop_pair, 2, column_size=self.config.char_in_word_size)
        hypothesis_char_vectors = fill_feature_vector_with_cropping_or_padding([batch[i]['sentence2_binary_parse_char_index'][:] for i in indices], hypothesis_pad_crop_pair, 2, column_size=self.config.char_in_word_size)

        premise_exact_match = construct_one_hot_feature_tensor([batch[i]["sentence1_token_exact_match_with_s2"][:] for i in indices], premise_pad_crop_pair, 1)
        hypothesis_exact_match = construct_one_hot_feature_tensor([batch[i]["sentence2_token_exact_match_with_s1"][:] for i in indices], hypothesis_pad_crop_pair, 1)    
        premise_exact_match = np.expand_dims(premise_exact_match, 2)
        hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)

        premise_inverse_term_frequency = hypothesis_inverse_term_frequency =  np.zeros((len(indices), self.config.seq_length,1))

        premise_antonym_feature = hypothesis_antonym_feature = premise_inverse_term_frequency

        premise_NER_feature = construct_one_hot_feature_tensor([batch[i]["sentence1_NER_feature"][:] for i in indices], premise_pad_crop_pair, 2, 7)
        hypothesis_NER_feature = construct_one_hot_feature_tensor([batch[i]["sentence2_NER_feature"][:] for i in indices], hypothesis_pad_crop_pair, 2, 7)

        return premise_vectors, hypothesis_vectors, labels, genres, premise_pos_vectors, \
                hypothesis_pos_vectors, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, hypothesis_inverse_term_frequency, \
                premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, hypothesis_NER_feature

    def get_em(self, example):
        s1_tokenize = example["sent_a"]
        s2_tokenize = example["sent_b"]

        premise_exact_match = []
        hypothesis_exact_match = []

        for idx, w1 in enumerate(s1_tokenize):
            for s2idx, w2 in enumerate(s2_tokenize):
                if (w1, w2) in self.em_cache:
                    mat = self.em_cache[(w1, w2)]
                else:
                    mat = self.em_cache[(w1, w2)] = is_exact_match(w1, w2)
                if mat:
                    premise_exact_match.append(idx)
                    hypothesis_exact_match.append(s2idx)

        example["sentence1_token_exact_match_with_s2"] = premise_exact_match
        example["sentence2_token_exact_match_with_s1"] = hypothesis_exact_match

        return example

    def step(self, batch, is_train=False, attack=False):
        if attack:
            start_time = time.time()
            sents = []
            for sample in batch:
                sample["sent_a"] = sample["sent_a"][:self.config.seq_length]
                sample["sent_b"] = sample["sent_b"][:self.config.seq_length]                
                sents.append(sample["sent_a"])
                sents.append(sample["sent_b"])
            pos = nltk.pos_tag_sents(sents)
            for i, sample in enumerate(batch):
                sample["sent_a_pos"] = pos[0]
                sample["sent_b_pos"] = pos[1]
                pos = pos[2:]
                sample["sentence1_binary_parse"] = " ".join(sample["sent_a"])
                sample["sentence2_binary_parse"] = " ".join(sample["sent_b"])
                sample["pairID"] = i
                sample["sentence1_part_of_speech_tagging"] = " ".join([w[1] for w in sample["sent_a_pos"]]) 
                sample["sentence2_part_of_speech_tagging"] = " ".join([w[1] for w in sample["sent_b_pos"]])
            batch = [self.get_em(sample) for sample in batch]
            with Pool(processes=8) as pool:
                batch = pool.map(get_ner, batch)                            
            for example in batch:
                for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                    example[sentence + '_index_sequence'] = np.zeros((self.FIXED_PARAMETERS["seq_length"]), dtype=np.int32)
                    example[sentence + '_inverse_term_frequency'] = np.zeros((self.FIXED_PARAMETERS["seq_length"]), dtype=np.float32)

                    if sentence == 'sentence1_binary_parse':
                        token_sequence = example["sent_a"]
                    else:
                        token_sequence = example["sent_b"]

                    padding = self.FIXED_PARAMETERS["seq_length"] - len(token_sequence)
                        
                    for i in range(self.FIXED_PARAMETERS["seq_length"]):
                        if i >= len(token_sequence):
                            index = self.word_indices[PADDING]
                            itf = 0
                        else:
                            if not (token_sequence[i] in self.word_indices):
                                index = 0
                            elif self.config.embedding_replacing_rare_word_with_UNK:
                                index = self.word_indices[token_sequence[i]] if self.word_counter[token_sequence[i]] >= self.config.UNK_threshold else self.word_indices["<UNK>"]
                            else:
                                index = self.word_indices[token_sequence[i]]
                        example[sentence + '_index_sequence'][i] = index
                        example[sentence + '_inverse_term_frequency'][i] = 0
                    
                    example[sentence + '_char_index'] = np.zeros((self.FIXED_PARAMETERS["seq_length"], self.config.char_in_word_size), dtype=np.int32)
                    for i in range(self.FIXED_PARAMETERS["seq_length"]):
                        if i >= len(token_sequence):
                            continue
                        else:
                            chars = [c for c in token_sequence[i]]
                            for j in range(self.config.char_in_word_size):
                                if j >= (len(chars)):
                                    break
                                else:
                                    if chars[j] in self.char_indices:
                                        index = self.char_indices[chars[j]]
                                    else:
                                        index = 0
                                example[sentence + '_char_index'][i,j] = index 

        start_time = time.time()

        minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres, \
        minibatch_pre_pos, minibatch_hyp_pos, pairIDs, premise_char_vectors, hypothesis_char_vectors, \
        premise_exact_match, hypothesis_exact_match, premise_inverse_term_frequency, \
        hypothesis_inverse_term_frequency, premise_antonym_feature, hypothesis_antonym_feature, premise_NER_feature, \
        hypothesis_NER_feature = \
            self.get_minibatch(batch, is_train)

        input_feed = {
            self.model.premise_x: minibatch_premise_vectors,
            self.model.hypothesis_x: minibatch_hypothesis_vectors,
            self.model.y: minibatch_labels, 
            self.model.keep_rate_ph: self.keep_rate if is_train else 1.0,
            self.model.is_train: is_train,
            self.model.premise_pos: minibatch_pre_pos,
            self.model.hypothesis_pos: minibatch_hyp_pos,
            self.model.premise_char:premise_char_vectors,
            self.model.hypothesis_char:hypothesis_char_vectors,
            self.model.premise_exact_match:premise_exact_match,
            self.model.hypothesis_exact_match: hypothesis_exact_match,
            self.model.premise_ner: premise_NER_feature,
            self.model.hypothesis_ner: hypothesis_NER_feature            
        }

        output_feed = [self.loss, self.accuracy, self.predictions, self.error]
        if is_train:
            output_feed.append(self.optimizer)

        res = self.sess.run(output_feed, input_feed)

        # print("inference time", time.time() - start_time)

        return res
