import math

import torch

sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}


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


class Instance(object):
    def __init__(self, sentence_pack, word2index, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.sentence_tokens = torch.zeros(args.max_sequence_len).long()

        '''generate sentence tokens'''
        words = self.sentence.split()
        self.length = len(words)
        for i, w in enumerate(words):
            # word = w.lower()
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']

        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()
        self.aspect_tags[self.length:] = -1
        self.opinion_tags[self.length:] = -1
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        self.tags[:, :] = -1

        for i in range(self.length):
            for j in range(i, self.length):
                self.tags[i][j] = 0
        for pair in sentence_pack['triples']:
            aspect = pair['target_tags']
            opinion = pair['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            for l, r in aspect_span:
                for i in range(l, r+1):
                    self.aspect_tags[i] = 1 if i == l else 2
                    self.tags[i][i] = 1
                    if i > l: self.tags[i-1][i] = 1
                    for j in range(i, r+1):
                        self.tags[i][j] = 1
            for l, r in opinion_span:
                for i in range(l, r+1):
                    self.opinion_tags[i] = 1 if i == l else 2
                    self.tags[i][i] = 2
                    if i > l: self.tags[i-1][i] = 2
                    for j in range(i, r+1):
                        self.tags[i][j] = 2
            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            if args.task == 'pair':
                                if i > j: self.tags[j][i] = 3
                                else: self.tags[i][j] = 3
                            elif args.task == 'triplet':
                                if i > j: self.tags[j][i] = sentiment2id[pair['sentiment']]
                                else: self.tags[i][j] = sentiment2id[pair['sentiment']]

        '''generate mask of the sentence'''
        self.mask = torch.zeros(args.max_sequence_len)
        self.mask[:self.length] = 1


def load_data_instances(sentence_packs, word2index, args):
    instances = list()
    for sentence_pack in sentence_packs:
        instances.append(Instance(sentence_pack, word2index, args))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentence_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentence_tokens.append(self.instances[i].sentence_tokens)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)

        indexes = list(range(len(sentence_tokens)))
        indexes = sorted(indexes, key=lambda x: lengths[x], reverse=True)

        sentence_ids = [sentence_ids[i] for i in indexes]
        sentence_tokens = torch.stack(sentence_tokens).to(self.args.device)[indexes]
        lengths = torch.tensor(lengths).to(self.args.device)[indexes]
        masks = torch.stack(masks).to(self.args.device)[indexes]
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)[indexes]
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)[indexes]
        tags = torch.stack(tags).to(self.args.device)[indexes]

        return sentence_ids, sentence_tokens, lengths, masks, aspect_tags, opinion_tags, tags
