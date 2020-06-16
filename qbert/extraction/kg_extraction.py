from __future__ import absolute_import, division, print_function

from albert import AlbertQA

import string

import collections

import torch

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils_squad import (get_predictions, read_squad_example,
                         convert_example_to_features, to_list, convert_examples_to_features, get_all_predictions)

import timeit
import torch.nn as nn

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])
conjunctions = ['and', 'or', 'nor']
articles = ["the", 'a', 'an', 'his', 'her', 'their', 'my', 'its', 'those', 'these', 'that', 'this', 'the']
pronouns = [" He ", " She ", " he ", " she ", " they ", " them "]


class Extraction(object):

    def __init__(self):
        self.albert = AlbertQA()

    def extract_entity(self, input_text, preds, probs, threshold=0.1, inv=False):

        entities = set()

        if preds is None:
            return []

        for pred, prob in zip(preds, probs):
            t = pred
            p = prob
            if len(t) < 1:
                continue
            if p > threshold and "MASK" not in t:
                for pred, prob in zip(preds, probs):
                    if t != pred and pred in t and prob > threshold and len(pred) > 2:
                        t = pred
                        p = prob
                        break
                t = t.strip(string.punctuation)
                remove = t

                # take out leading articles for cleaning
                words = t.split()
                if len(words) == 0:
                    break
                if words[0].lower() in articles:
                    remove = " ".join(words[1:])
                    words[0] = words[0].lower()
                    t = " ".join(words[1:])
                if ',' in t:
                    t = t.split(',')
                    entities.update(t)
                    t = t[0]
                else:
                    entities.add(t)

                if 'empty' in t and inv:
                    return []

                input_text = input_text.replace(remove, '[MASK]').replace('  ', ' ').replace(' .', '.')
        return list(entities)

    def generate(self, input_text, threshold=0.2, attribute=True):
        input_text = input_text
        locs = []
        objs_surr = []
        objs_inv = []
        primers = ["Where am I located?", "What is here?", "Which objects are in my inventory?"]

        res = self.albert.batch_predict_top_k([input_text] * 3, primers, 10)

        for r, v in res.items():
            preds = [a['text'] for a in v]
            probs = [a['probability'] for a in v]
            if r == 0:
                locs = self.extract_entity(input_text, preds, probs, threshold)
            if r == 1:
                objs_surr = self.extract_entity(input_text, preds, probs, threshold/10)
            if r == 2:
                objs_inv = self.extract_entity(input_text, preds, probs, threshold/10, True)

        objs = objs_surr + objs_inv
        primers = ["What attribute does " + o + " have?" for o in objs]
        attributes = {o: [] for o in objs_surr + objs_inv}
        if attribute:
            res = self.albert.batch_predict_top_k([input_text] * len(objs), primers, 10)

            for r, v in res.items():
                preds = [a['text'] for a in v]
                probs = [a['probability'] for a in v]
                attributes[objs[r]] += self.extract_entity(input_text, preds, probs, threshold/10)

        return {'location': locs, 'object_surr': objs_surr, 'objs_inv': objs_inv, 'attributes': attributes}
