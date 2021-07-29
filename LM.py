import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class LM:
    def __init__(self, args):
        self.max_sent_length = args.max_sent_length
        self.bert_model = args.bert_model
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForMaskedLM.from_pretrained(self.bert_model)
        self.model.to(self.device)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.lm_candidates = args.lm_candidates
        print("BertForMaskedLM loaded")

    def get_pred(self, prefixes, suffixes):
        tokens, segments, masks = [], [], []

        for i in range(len(prefixes)):
            prefix, suffix = prefixes[i], suffixes[i]
            tokens_prefix = self.tokenizer.tokenize(" ".join(prefix))
            tokens_suffix = self.tokenizer.tokenize(" ".join(suffix))
            tokens_prefix = tokens_prefix[len(tokens_prefix) - (self.max_sent_length - 3) // 2:]
            tokens_suffix = tokens_suffix[:(self.max_sent_length - 3) // 2]
            tokenized_text = ["[CLS]"] + tokens_prefix + ["[MASK]"] + tokens_suffix + ["[SEP]"]
            mask_id = 0
            while tokenized_text[mask_id] != "[MASK]": mask_id += 1
            masks.append(mask_id)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            indexed_tokens += [0] * (self.max_sent_length - len(indexed_tokens))
            tokens.append(indexed_tokens)
       
        tokens_tensor = torch.tensor(tokens).cuda()
        segments_tensor = torch.zeros((len(prefixes), self.max_sent_length), dtype=torch.long).cuda()

        with torch.no_grad():
            scores = self.model(tokens_tensor, segments_tensor)
            pred = []
            for i in range(len(prefixes)):
                pred.append(self.softmax(scores[i, masks[i], :]))
            return pred

    def query(self, prefix, word_list, suffix):
        predictions = self.get_pred(prefix, suffix)
        tokenized_text = self.tokenizer.tokenize(" ".join(word_list))
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        prob = [predictions[idx] for idx in indexed_tokens]
        return prob

    def query_topk(self, prefix_a, suffix_a, prefix_b, suffix_b):
        pred_a = self.get_pred(prefix_a, suffix_a)
        pred_b = self.get_pred(prefix_b, suffix_b)
        predicted_index = torch.topk(pred_a * pred_b, self.lm_candidates)[1]
        res = self.tokenizer.convert_ids_to_tokens([idx.item() for idx in predicted_index])
        return res
