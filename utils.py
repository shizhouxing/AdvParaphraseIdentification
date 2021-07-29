import numpy as np
import json, re, os, nltk, pickle, gzip, random
from tqdm import tqdm
from multiprocessing import Pool
from vocab import Vocab
from models.DIIN.util.data_processing import is_exact_match

if not os.path.exists('cache'): os.mkdir('cache')   

def save_json(filename, data):
    with open(filename, 'w') as file:
        file.write(json.dumps(data))

def tokenize(sent):
    return nltk.word_tokenize(sent)

def parse_line_quora(line):
    l = line.split('\t')
    example = {
        'sent_a': tokenize(l[1]),
        'sent_b': tokenize(l[2]),
        'label': int(l[0])
    }
    example['sent_a_pos'] = nltk.pos_tag([w.lower() for w in example['sent_a']])
    example['sent_b_pos'] = nltk.pos_tag([w.lower() for w in example['sent_b']])
    return example  

def parse_line_mrpc(line):
    if (line[-1] == '\n'): line = line[:-1]
    l = line.split('\t')
    example = {
        'sent_a': tokenize(l[3]),
        'sent_b': tokenize(l[4]),
        'label': int(l[0])
    }
    example['sent_a_pos'] = nltk.pos_tag([w.lower() for w in example['sent_a']])
    example['sent_b_pos'] = nltk.pos_tag([w.lower() for w in example['sent_b']])
    return example      

def build_vocab(args, data):
    path = 'cache/vocab_%s_%s.pkl' % (args.data, args.target)
    if os.path.exists(path):
        print('Loading cached vocabulary...')
        with open(path, 'rb') as file:
            vocab_char, vocab_word = pickle.load(file)
        return vocab_char, vocab_word

    print('Building vocabulary...')
    vocab = {}
    all_sents = []   
    for q in data:
        if args.task == 'classification':
            all_sents.append(q['sent_a'])
            all_sents.append(q['sent_b'])
        elif args.task == 'ranking':
            all_sents.append(q['query'])
            for cand in q['candidates']:
                all_sents.append(cand['sent'])
        else:
            raise NotImplementedError

    for sent in all_sents:
        for token in sent:
            t = token.lower()
            if t in vocab:
                vocab[t] += 1
            else:
                vocab[t] = 1
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)

    vocab_char = Vocab(args.dim_char_embed, args, random_init=True)
    vocab_word = Vocab(args.dim_word, args)

    print('Loading word vectors...')
    vectors = {}
    if not args.debug:
        with open(args.word_vector) as f_in:
            for line in f_in:
                line = line.split(' ')
                vectors[line[0]] = list(map(float, line[1:]))
    cnt_pretrained = 0
    for i, word in enumerate(vocab_list):
        for c in word: 
            if not c in vocab_char.symbol2index: 
                vocab_char.add(c)
            
        if word in vectors:
            vocab_word.add(word, vectors[word])
            cnt_pretrained += 1
        else:
            vocab_word.add(word)
    print('Pre-trained word vectors: %d/%d' % (cnt_pretrained, vocab_word.size))
    print('Number of chars:', vocab_char.size)

    with open(path, 'wb') as file:
        pickle.dump((vocab_char, vocab_word), file)

    return vocab_char, vocab_word

class PBar:
    def __init__(self):
        pass

    def set_total(self, tot):
        self.bar = tqdm(total=tot)
    
    def update(self):
        self.bar.update(1)

pbar = PBar()    

def generate_exact_match_features(example):
    em_a = np.zeros(len(example['sent_a']))
    em_b = np.zeros(len(example['sent_b']))

    for i, w1 in enumerate(example['sent_a']):
        for j, w2 in enumerate(example['sent_b']):
            if is_exact_match(w1, w2):
                em_a[i] = em_b[j] = 1

    example['em_a'], example['em_b'] = em_a, em_b
    pbar.update()
    return example

def symbol2index(args, data, vocab_char, vocab_word):
    def get_idx(sent):
        if len(sent) > args.max_sent_length:
            sent = sent[:args.max_sent_length]
        char_idx, word_idx = [], []
        for word in sent:
            idx = []
            for c in word:
                if len(idx) == args.max_char_per_word: break
                idx.append(vocab_char.get(c))
            char_idx.append(idx)
            word_idx.append(vocab_word.get(word))
        if len(word_idx) < args.max_sent_length:
            char_idx.append([vocab_char.EOS_ID]) 
            word_idx.append(vocab_word.EOS_ID)

        return char_idx, word_idx

    for example in data:
        example['char_idx_a'], example['word_idx_a'] = get_idx(example['sent_a'])
        example['char_idx_b'], example['word_idx_b'] = get_idx(example['sent_b'])

    if args.use_exact_match:
        print('Generating exact-match features...')
        pbar.set_total(len(data) // args.cpus)
        with Pool(processes=args.cpus) as pool:
            data = pool.map(generate_exact_match_features, data)
    
    return data

def load_data_raw(args, set):
    path = 'data/%s/%s.tsv' % (args.data, set)
    print('Loading data from ' + path)
    inp = []
    with open(path) as file:
        for line in file.readlines():
            if line[0] != '-':
                inp.append(line)

    if args.data == 'qqp' or args.data == 'paws_qqp':
        with Pool(processes=args.cpus) as pool:
            data = pool.map(parse_line_quora, inp)
    elif args.data == 'mrpc':
        data = [parse_line_mrpc(line) for line in inp[1:]] 
    else:
        raise NotImplementedError

    return data

def load_paws_test(args):
    assert(args.data == 'qqp')
    path = 'data/paws_qqp/test.tsv'
    print('Loading data from ' + path)
    inp = []
    with open(path) as file:
        for line in file.readlines():
            if line[0] != '-':
                inp.append(line)

        with Pool(processes=args.cpus) as pool:
            data = pool.map(parse_line_quora, inp)

    return data

def load_data(args):
    path_train = 'cache/data_{}_train.pkl'.format(args.data)
    path_valid = 'cache/data_{}_valid.pkl'.format(args.data)
    path_test = 'cache/data_{}_test.pkl'.format(args.data)
    path_vocab = 'cache/vocab_{}.pkl'.format(args.data)
    if os.path.exists(path_train) and os.path.exists(path_valid) and \
            os.path.exists(path_test) and os.path.exists(path_vocab):
        print('Loading cached common data...')
        if args.pre_train or args.adv_train:
            with open(path_train, 'rb') as file: data_train = pickle.load(file)
        else:
            data_train = []
        with open(path_valid, 'rb') as file: data_valid = pickle.load(file)
        with open(path_test, 'rb') as file: data_test = pickle.load(file)
        with open(path_vocab, 'rb') as file: vocab_char, vocab_word = pickle.load(file)
    else:
        data_train = load_data_raw(args, 'train')
        vocab_char, vocab_word = build_vocab(args, data_train)
        data_train = symbol2index(args, data_train, vocab_char, vocab_word)
        data_valid = symbol2index(args, load_data_raw(args, 'dev'), vocab_char, vocab_word)
        data_test = symbol2index(args, load_data_raw(args, 'test'), vocab_char, vocab_word)
        with open(path_train, 'wb') as file: pickle.dump(data_train, file)
        with open(path_valid, 'wb') as file: pickle.dump(data_valid, file)
        with open(path_test, 'wb') as file: pickle.dump(data_test, file)
        with open(path_vocab, 'wb') as file: pickle.dump((vocab_char, vocab_word), file)
    return data_train, data_valid, data_test, vocab_char, vocab_word

def get_batches(args, data, batch_size):
    batches = []
    for i in range((len(data) + batch_size - 1) // batch_size):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches

def sample_data(args, data):
    from tools.dump import write_for_mturk
    sampled = []
    for _ in range(args.sample_num):
        while True:
            example = data[random.randint(0, len(data) - 1)]
            if example['label'] == args.sample_label:
                sampled.append({
                    'adv': 0,
                    'label': example['label'],
                    's1': ' '.join(example['sent_a']).lower(),
                    's2': ' '.join(example['sent_b']).lower()
                })
                break
    if not os.path.exists(args.sample_output):
        os.mkdir(args.sample_output)
    write_for_mturk(args.sample_output, sampled)
    print('Wrote to {}'.format(args.sample_output))
