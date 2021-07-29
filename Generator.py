import nltk, random, time, json
import numpy as np
import torch
from utils import symbol2index, get_batches
from scipy.special import softmax
from multiprocessing import Pool
from LM import LM

class Generator:
    def __init__(self, args, target, vocab_char, vocab_word, data_train):
        self.args = args
        self.cpus = args.cpus
        self.vocab_char = vocab_char
        self.vocab_word = vocab_word
        self.data = data
        self.target = target
        self.task = args.task
        self.advgen_batch_size = args.advgen_batch_size
        self.max_num_steps = args.max_num_steps
        self.max_num_candidates = args.max_num_candidates
        self.max_sent_length_diff = args.max_sent_length_diff
        self.lm_batch_size = args.lm_batch_size
        self.batch_size = args.batch_size
        self.adv_file = args.adv_file
        self.lm_gram = args.lm_gram
        self.lm_candidates = args.lm_candidates
        self.beam_size = args.beam_size
        self.first_attack = args.first_attack
        self.query_batch_size = args.query_batch_size
        self.sample_only = args.sample_only
        self.max_num_candidates = self.lm_candidates
        self.PAD = '[PAD]'        
        self.lm = LM(args)
        self.vocab_word.build_stopwords()

    def _evaluate(self, data, verbose, label):
        num = len(data)

        cnt = 0
        start_time = time.time()
        adv = []
        for t, (sent_a, sent_b) in enumerate(data):
            res = self.query(sent_a, sent_b, label)

            if verbose:
                print('#%d/%d (%.4fs/example)' % (
                    t + 1, num, (time.time() - start_time) * 1. / (t + 1)))
            if res is not None:
                cnt += 1
                if verbose:
                    print('  OK')
                    print(' ', ' '.join(sent_a))
                    print(' ', ' '.join(sent_b))
                    print(' ', ' '.join(res['sent_a']))
                    print(' ', ' '.join(res['sent_b']))
                    adv.append({
                        'sent_a': ' '.join(res['sent_a']),
                        'sent_b': ' '.join(res['sent_b']),
                        'sent_a_original': ' '.join(sent_a),
                        'sent_b_original': ' '.join(sent_b),
                        'label': label
                    })
            else:
                if verbose:
                    print('  Failed')
                    print(' ', ' '.join(sent_a))
                    print(' ', ' '.join(sent_b))   
            if verbose:
                print('Success rate (label=%d): %.5f' % (label, cnt * 1. / (t + 1)))

            if (t + 1) % 10 == 0:
                with open('tmp-%d-%s' % (label, self.adv_file), 'w') as file:
                    file.write(json.dumps({
                        'adv': adv,
                        'suc_rate': cnt * 1. / (t + 1),
                        'steps': t + 1
                    }))

        return cnt * 1. / num, adv

    def sample(self, data, num, label):
        len_data = len(data)
        sampled = []
        
        cnt = 0
        start_time = time.time()
        for t in range(num):
            if label == 1:
                while True:
                    i = random.randint(0, len_data - 1)
                    if data[i]['label'] == 1:
                        sent_a = data[i]['sent_a']
                        sent_b = data[i]['sent_b']
                        break
            elif label == 0:
                while True:
                    i = random.randint(0, len_data - 1)
                    j = random.randint(0, len_data - 1)
                    sent_a = data[i]['sent_a']
                    sent_b = data[j]['sent_b']
                    c = 0
                    while i == j or \
                            abs(len(sent_b) - len(sent_a)) > self.max_sent_length_diff or\
                            sent_a[0].lower() == sent_b[0].lower():
                        j = random.randint(0, len_data - 1)
                        sent_b = data[j]['sent_b']
                        c += 1
                        if c >= 100: break
                    if c < 100: break

            sampled.append((sent_a, sent_b))

        return sampled

    def evaluate(self, data, num, verbose=False):
        sampled_pos = self.sample(data, num, 1)
        sampled_neg = self.sample(data, num, 0)

        if self.sample_only:
            data_pos, data_neg = [], []
            for t, (sent_a, sent_b) in enumerate(sampled_pos):
                data_pos.append({
                    'sent_a': sent_a,
                    'sent_b': sent_b,
                    'label': 1
                })
            for t, (sent_a, sent_b) in enumerate(sampled_neg):
                data_neg.append({
                    'sent_a': sent_a,
                    'sent_b': sent_b,
                    'label': 0
                })                

            pred = self.target.step(
                symbol2index(self.args, data_pos, self.vocab_char, self.vocab_word), attack=True)[2]
            suc_pos, adv_pos = 0., []
            for i in range(len(pred)):
                if pred[i] == 0: 
                    suc_pos += 1
                    adv_pos.append(data_pos[i])
            suc_pos = suc_pos * 1. / len(pred)                    

            pred = self.target.step(
                symbol2index(self.args, data_neg, self.vocab_char, self.vocab_word), attack=True)[2]
            suc_neg, adv_neg = 0., []
            for i in range(len(pred)):
                if pred[i] == 1: 
                    suc_neg += 1
                    adv_neg.append(data_neg[i])                              
            suc_neg = suc_neg * 1. / len(pred)
        elif self.first_attack == 1:
            suc_pos, adv_pos = self._evaluate(sampled_pos, verbose, 1)
            suc_neg, adv_neg = self._evaluate(sampled_neg, verbose, 0)
        else:
            suc_neg, adv_neg = self._evaluate(sampled_neg, verbose, 0)
            suc_pos, adv_pos = self._evaluate(sampled_pos, verbose, 1)

        if verbose:
            print('Success rate (pos) : %.5f' % suc_pos)
            print('Success rate (neg) : %.5f' % suc_neg)         
        
        with open(self.adv_file, 'w') as file:
            file.write(json.dumps({
                'adv_pos': adv_pos,
                'adv_neg': adv_neg,
                'suc_pos': suc_pos,
                'suc_neg': suc_neg
            }))

        return suc_pos, suc_neg    

    def query(self, sent_a_list, sent_b_list, label):
        non_batch = (type(sent_a_list[0]) != type([]))
        if non_batch:
            sent_a_list, sent_b_list = [sent_a_list], [sent_b_list]

        n = len(sent_a_list)
        examples = [
            { 'sent_a': sent_a_list[i], 'sent_b': sent_b_list[i], 'label': label, 'src': i }\
            for i in range(n)
        ]
        res = [None] * n

        r = self.target.step(
            symbol2index(self.args, examples, self.vocab_char, self.vocab_word), attack=True)[2]
        for i in range(n):
            if r[i] != label:
                res[i] = examples[i]

        examples = [[example] for example in examples]

        sum_time_lm = 0.
        sum_time_target = 0.

        for t in range(self.max_num_steps):
            num_candidates = [0] * n
            candidates = []

            sent = []
            for k in range(n):
                if res[k] is not None: continue
                if len(examples[k]) == 0: continue   

                for bid, example in enumerate(examples[k]):
                    sent.append([w.lower() for w in example['sent_a']])
                    sent.append([w.lower() for w in example['sent_b']])
            pos = nltk.pos_tag_sents(sent)            

            dedup = {}

            for k in range(n):
                if res[k] is not None: continue
                if len(examples[k]) == 0: continue   

                for bid, example in enumerate(examples[k]):
                    pos_a = pos[0]
                    pos_b = pos[1]
                    pos = pos[2:]
                    cand = []
                    for i in range(len(pos_a)):
                        for j in range(len(pos_b)):
                            if not (pos_a[i][0].lower() in self.vocab_word.stopwords) and \
                                    not (pos_b[j][0].lower() in self.vocab_word.stopwords) and \
                                    pos_a[i][1][0] in ['N', 'V', 'J'] and \
                                    pos_b[j][1][0] in ['N', 'V', 'J'] and \
                                    pos_a[i][1][0] == pos_b[j][1][0] and \
                                    (label == 0 or pos_a[i][0] == pos_b[j][0]):
                                sent_a = example['sent_a'][:i] + \
                                    [self.PAD] + example['sent_a'][i+1:]
                                sent_b = example['sent_b'][:j] + \
                                    [self.PAD] + example['sent_b'][j+1:]
                                sent_a_join = ' '.join(sent_a)
                                sent_b_join = ' '.join(sent_b)
                                if (sent_a_join, sent_b_join) in dedup:
                                    continue
                                dedup[(sent_a_join, sent_b_join)] = 1
                                cand.append({
                                    'sent_a': sent_a,
                                    'sent_b': sent_b,
                                    'label': label,
                                    'src': example['src']
                                })  

                    if len(cand) > self.max_num_candidates:
                        random.shuffle(cand)
                        cand = cand[:self.max_num_candidates]
                    num_candidates[k] += len(cand)
                    candidates += cand

            if len(candidates) == 0: 
                if non_batch: res = res[0]
                return res
            candidates = symbol2index(self.args, candidates, self.vocab_char, self.vocab_word)
            start_time = time.time()

            pred, error = [], []

            for batch in get_batches(self.args, candidates, self.query_batch_size):
                r = self.target.step(batch, attack=True)
                pred += list(r[2])
                error += list(r[3])

            sum_time_target += time.time() - start_time
            
            if label == 1:
                for i in range(len(candidates)):
                    error[i] = 1 - error[i]
            rank = sorted(range(len(candidates)), key=lambda idx: error[idx], reverse=True)
  
            for k in range(n):
                if res[k] is not None: continue
                if len(examples[k]) == 0: continue     
                examples[k] = [candidates[idx] for idx in rank[:self.beam_size]]
                rank = rank[self.beam_size:]

            num_candidates = [0] * n
            candidates = []
            prefixes, suffixes = [], []
            dedup = {}

            for k in range(n):
                if res[k] is not None: continue
                if len(examples[k]) == 0: continue
                for bid, example in enumerate(examples[k]):
                    idx_i, idx_j = 0, 0
                    while example['sent_a'][idx_i] != self.PAD: idx_i += 1
                    while example['sent_b'][idx_j] != self.PAD: idx_j += 1
                    i, j = idx_i, idx_j
                    prefix_a = example['sent_a'][max(0, i-self.lm_gram):i]
                    suffix_a = example['sent_a'][i+1:i+self.lm_gram+1]   
                    prefixes.append(prefix_a)
                    suffixes.append(suffix_a)                    
                    prefix_b = example['sent_b'][max(0, j-self.lm_gram):j]
                    suffix_b = example['sent_b'][j+1:j+self.lm_gram+1]                                
                    prefixes.append(prefix_b)
                    suffixes.append(suffix_b)

            lm_pred = []
            for i in range((len(prefixes) + self.lm_batch_size - 1) // self.lm_batch_size):
                p = prefixes[i * self.lm_batch_size : (i + 1) * self.lm_batch_size]
                s = suffixes[i * self.lm_batch_size : (i + 1) * self.lm_batch_size]
                for pred in self.lm.get_pred(p, s):
                    lm_pred.append(pred.reshape((1, -1)))
            lm_pred = torch.cat(lm_pred, dim=0)


            for k in range(n):
                if res[k] is not None: continue
                if len(examples[k]) == 0: continue
                    
                for bid, example in enumerate(examples[k]):
                    idx_i, idx_j = 0, 0
                    while example['sent_a'][idx_i] != self.PAD: idx_i += 1
                    while example['sent_b'][idx_j] != self.PAD: idx_j += 1
                    cand = []
                    start_time = time.time()
                    i, j = idx_i, idx_j
                    pred_a = lm_pred[0]
                    pred_b = lm_pred[1]
                    lm_pred = lm_pred[2:]
                    cnt_a, cnt_b = {}, {}
                    for w in example['sent_a']: 
                        if not w.lower() in cnt_a: cnt_a[w.lower()] = 1
                        else: cnt_a[w.lower()] += 1
                    for w in example['sent_b']: 
                        if not w.lower() in cnt_b: cnt_b[w.lower()] = 1
                        else: cnt_b[w.lower()] += 1
                    cnt_a[example['sent_a'][idx_i].lower()] -= 1
                    cnt_b[example['sent_b'][idx_j].lower()] -= 1

                    predicted_index = torch.topk(pred_a * pred_b, self.lm_candidates)[1]
                    cand_words = self.lm.tokenizer.convert_ids_to_tokens([idx.item() for idx in predicted_index])
                    for w in cand_words:
                        if w in self.vocab_word.symbol2index and \
                                not (w in self.vocab_word.stopwords) and \
                                (not (w in cnt_a) or (cnt_a[w] == 0)) and \
                                (not (w in cnt_b) or (cnt_b[w] == 0)):
                            sent_a = example['sent_a'][:i] + [w] + example['sent_a'][i+1:]
                            sent_b = example['sent_b'][:j] + [w] + example['sent_b'][j+1:]
                            sent_a_join = ' '.join(sent_a)
                            sent_b_join = ' '.join(sent_b)
                            if (sent_a_join, sent_b_join) in dedup:
                                continue
                            dedup[(sent_a_join, sent_b_join)] = 1                                
                            cand.append({
                                'sent_a': sent_a,
                                'sent_b': sent_b,
                                'label': label,
                                'src': example['src']
                            })  

                    if len(cand) > self.max_num_candidates:
                        random.shuffle(cand)
                        cand = cand[:self.max_num_candidates]
                    num_candidates[k] += len(cand)                        
                    candidates += cand

            assert(len(lm_pred) == 0) 

            if len(candidates) == 0: 
                if non_batch: res = res[0]
                return res
            candidates = symbol2index(self.args, candidates, self.vocab_char, self.vocab_word)
            start_time = time.time()

            pred, error = [], []
            for batch in get_batches(self.args, candidates, self.query_batch_size):
                r = self.target.step(batch, attack=True)
                pred += list(r[2])
                error += list(r[3])

            sum_time_target += time.time() - start_time

            for k in range(n):
                if res[k] is not None: continue
                if num_candidates[k] == 0:
                    examples[k] = []                
                if len(examples[k]) == 0: continue
                best, opt = -1, -1
                for i in range(num_candidates[k]):
                    if label == 1: 
                        error[i] = 1 - error[i]
                rank = sorted(range(num_candidates[k]), key=lambda idx: error[idx], reverse=True)
                if pred[rank[0]] != label:
                    res[k] = candidates[rank[0]]
                else:
                    examples[k] = [candidates[idx] for idx in rank[:self.beam_size]]
                candidates = candidates[num_candidates[k]:]
                pred = pred[num_candidates[k]:]
                error = error[num_candidates[k]:]     
               
        if non_batch: res = res[0]

        return res

    def generate_batch(self, label=0, verbose=False):
        start_time = time.time()
        sampled = self.sample(self.data, self.advgen_batch_size, label)
        sent_a_list = [example[0] for example in sampled]
        sent_b_list = [example[1] for example in sampled]
        res = self.query(sent_a_list, sent_b_list, label)
        data = []
        for i in range(len(res)):
            if res[i] is not None:
                data.append(res[i])
        if len(data) > 0:
            print('successful')
        else:
            print('failed')
        suc_rate = len(data) * 1. / len(res)
        return data, suc_rate