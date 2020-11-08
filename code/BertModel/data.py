import math

import torch
import numpy as np

sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
from transformers import BertTokenizer


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(tags, length, token_range):
    '''for BIO tag'''
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    def __init__(self, tokenizer, sentence_pack, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.tokens = self.sentence.strip().split()
        self.sen_length = len(self.tokens)
        self.token_range = []
        self.bert_tokens = tokenizer.encode(self.sentence)
        self.length = len(self.bert_tokens)
        self.bert_tokens_padding = torch.zeros(args.max_sequence_len).long()
        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.mask = torch.zeros(args.max_sequence_len)

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        self.mask[:self.length] = 1

        token_start = 1
        for i, w, in enumerate(self.tokens):
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        assert self.length == self.token_range[-1][-1]+2

        self.aspect_tags[self.length:] = -1
        self.aspect_tags[0] = -1
        self.aspect_tags[self.length-1] = -1

        self.opinion_tags[self.length:] = -1
        self.opinion_tags[0] = -1
        self.opinion_tags[self.length - 1] = -1

        self.tags[:, :] = -1
        for i in range(1, self.length-1):
            for j in range(i, self.length-1):
                self.tags[i][j] = 0

        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            '''set tag for aspect'''
            for l, r in aspect_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        self.tags[i][j] = 1
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    al, ar = self.token_range[i]
                    self.aspect_tags[al] = set_tag
                    self.aspect_tags[al+1:ar+1] = -1
                    '''mask positions of sub words'''
                    self.tags[al+1:ar+1, :] = -1
                    self.tags[:, al+1:ar+1] = -1

            '''set tag for opinion'''
            for l, r in opinion_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        self.tags[i][j] = 2
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    pl, pr = self.token_range[i]
                    self.opinion_tags[pl] = set_tag
                    self.opinion_tags[pl+1:pr+1] = -1
                    self.tags[pl+1:pr+1, :] = -1
                    self.tags[:, pl+1:pr+1] = -1

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal:sar+1, spl:spr+1] = -1
                            if args.task == 'pair':
                                if i > j:
                                    self.tags[spl][sal] = 3
                                else:
                                    self.tags[sal][spl] = 3
                            elif args.task == 'triplet':
                                if i > j:
                                    self.tags[spl][sal] = sentiment2id[triple['sentiment']]
                                else:
                                    self.tags[sal][spl] = sentiment2id[triple['sentiment']]


def load_data_instances(sentence_packs, args):
    instances = list()
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    for sentence_pack in sentence_packs:
        instances.append(Instance(tokenizer, sentence_pack, args))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)
        return sentence_ids, bert_tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags
