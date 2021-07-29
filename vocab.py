import tensorflow as tf
import numpy as np
import nltk
from multiprocessing import Pool

class Vocab:
    def __init__(self, dim_embed, args, random_init=False):
        self.cpus = args.cpus
        self.random_init = random_init
        self.size = 0
        self.symbols = []
        self.symbol2index = {}
        self.dim_embed = dim_embed
        self.embed = []
        self.simwords = []

        self.build_stopwords()

        self.PAD_ID = 0
        self.add("PAD")
        self.UNK_ID = 1
        self.add("UNK")
        self.EOS_ID = 2
        self.add("EOS")
        self.GO_ID = 3
        self.add("GO")
    
    def add(self, symbol, embed=None):
        self.symbols.append(symbol)
        self.symbol2index[symbol] = self.size
        if embed is None:
            if self.random_init:
                self.embed.append(np.random.uniform(low=-0.05, high=0.05, size=self.dim_embed))
            else:
                self.embed.append(np.zeros(self.dim_embed, dtype=np.float32))
        else:
            self.embed.append(embed)
        self.size += 1      

    def get(self, symbol):
        symbol = symbol.lower() 
        if symbol in self.symbol2index:
            return self.symbol2index[symbol]
        else:
            return self.UNK_ID      

    def build_embed(self, name, trainable=True):
        self.build_stopwords() # TODO: to be removed
        self.embed_init = np.array(self.embed, dtype=np.float32)
        with tf.variable_scope("embedding"):
            embed = tf.get_variable(
                name, dtype=tf.float32, 
                initializer=self.embed_init, trainable=trainable)

    def build_stopwords(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        stopwords = ["?", ",", "af", "``", "and", ":", ".", "...", "thing", 
            "ya", "-", ";", '\'', "ye", "one", "things", "`", "b", "us", "something"
            "nothin", "nothing", "none", "either", "-lrb-", "-rrb-", "even", "/", '[', 
            ']', "\""]
        for w in stopwords:
            self.stopwords.add(w)

    def build_candwords(self, data):
        self.candwords = {}
        for example in data:
            for w in example["sent_a_pos"] + example["sent_b_pos"]:
                if w[1][0] in ['N', 'V', 'J']:
                    self.candwords[w[0]] = True
