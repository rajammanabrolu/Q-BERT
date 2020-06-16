import os
import networkx as nx
import numpy as np
from fuzzywuzzy import fuzz
from jericho.util import clean
from jericho import defines
import extraction_api
#from extraction import kgextraction


class StateAction(object):

    def __init__(self, spm, vocab, vocab_rev, tsv_file, max_word_len, askbert, attr_file):
        self.graph_state = nx.DiGraph()
        self.max_word_len = max_word_len
        self.graph_state_rep = []
        self.visible_state = ""
        self.drqa_input = ""
        self.vis_pruned_actions = []
        self.pruned_actions_rep = []
        self.sp = spm
        self.vocab_act = vocab
        self.vocab_act_rev = vocab_rev
        vocabs = self.load_vocab_kge(tsv_file)
        self.vocab_kge, self.vocab_kgr = vocabs['entity'], vocabs['relation']
        self.context_attr = self.load_attributes(attr_file)
        #print(self.context_attr)
        self.adj_matrix = np.array([np.zeros((len(self.vocab_kge), len(self.vocab_kge)))] * len(self.vocab_kgr))
        self.room = ""
        self.askbert = askbert
        self.ABBRV_DICT = {
            'n': 'north',
            's': 'south',
            'w': 'west',
            'e': 'east',
            'd': 'down',
            'u': 'up',
            'g': 'again',
            'ne': 'northeast',
            'nw': 'northwest',
            'se': 'southeast',
            'sw': 'southwest',
        }
        self.MOVE_ACTIONS = 'north/south/west/east/northwest/southwest/northeast/southeast/up/down/enter/exit'.split('/')
        askbert_args = {'input_text': '', 'length': 10, 'batch_size': 1, 'temperature': 1, 'model_name': '117M',
                        'seed': 0, 'nsamples': 10, 'cutoffs': "6 7 5", 'write_sfdp': False, 'random': False}
        #self.extraction = kgextraction.World(askbert_args)
        self.ct = 0

    def visualize(self):
        import matplotlib.pyplot as plt
        pos = nx.kamada_kawai_layout(self.graph_state)
        edge_labels = {e: self.graph_state.edges[e]['rel'] for e in self.graph_state.edges}
        triples = []
        for k, v in edge_labels.items():
            triples.append((k[0], v, k[1]))
        print(triples)
        #print(len(edge_labels.keys()), edge_labels)
        nx.draw_networkx_edge_labels(self.graph_state, pos, edge_labels)
        nx.draw(self.graph_state, pos=pos, with_labels=True, node_size=2000, font_size=10)
        plt.savefig(str(self.ct) + '.pdf')
        self.ct += 1
        # plt.show()

    def load_vocab_kge(self, tsv_file):
        ent = {}
        #alle = []
        with open(tsv_file, 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[str(e.strip())] = int(eid.strip())
                #alle.append(str(e.strip()))
        #print(len(ent), len(alle), ent.keys(), alle)
        rel_path = os.path.dirname(tsv_file)
        rel_name = os.path.join(rel_path, 'relation2id.tsv')
        rel = {}
        with open(rel_name, 'r') as f:
            for line in f:
                r, rid = line.split('\t')
                rel[r.strip()] = int(rid.strip())
        return {'entity': ent, 'relation': rel}

    def load_attributes(self, attr_file):
        context_attr = ""
        attr_file = './attrs/' + attr_file + '_attr.txt'
        if os.path.isfile(attr_file):
            with open(attr_file, 'r') as f:
                context_attr = str(f.read())
        context_attr = "talkable, seen, lieable, enterable, nodwarf, indoors, visited, handed, lockable, surface, thing, " \
                           "water_room, unlock, lost, afflicted, is_treasure, converse, mentioned, male, npcworn, no_article, " \
                           "relevant, scored, queryable, town, pluggable, happy, is_followable, legible, multitude, burning, " \
                           "room, clothing, underneath, ward_area, little, intact, animate, bled_in, supporter, readable, " \
                           "openable, near, nonlocal, door, plugged, sittable, toolbit, vehicle, light, lens_searchable, " \
                           "open, familiar, is_scroll, aimable, takeable, static, unique, concealed, vowelstart, alcoholic, " \
                           "bodypart, general, is_spell, full, dry_land, pushable, known, proper, inside, clean, " \
                           "ambiguously_plural, container, edible, treasure, can_plug, weapon, is_arrow, insubstantial, " \
                           "pluralname, transparent, is_coin, air_room, scenery, on, is_spell_book, burnt, burnable, " \
                           "auto_searched, locked, switchable, absent, rockable, beenunlocked, progressing, severed, worn, " \
                           "windy, stone, random, neuter, legible, female, asleep, wiped"

        return context_attr

    def update_state(self, visible_state, inventory_state, objs, prev_act=None, cache=None):
        self.visible_state = visible_state
        prev_room = self.room
        add_triples = set()
        remove_triples = set()
        add_triples.add(('you', 'is', 'you'))

        if cache is not None:
            entities = cache
        else:
            entities = extraction_api.call_askbert(self.visible_state + ' [atr] ' + self.context_attr,
                                                   self.askbert, self.context_attr != "")
            if entities is None:
                self.askbert /= 1.5
                return [], None
            entities = entities['entities']

        # Location mappings
        if len(entities['location']) != 0:
            self.room = entities['location'][0]

        if len(entities['location']) == 0:
            self.room = ""

        if self.room != "":
            add_triples.add(('you', 'in', self.room))
            remove_triples.add(('you', 'in', prev_room))

            if prev_act.lower() in self.MOVE_ACTIONS:
                add_triples.add((self.room, prev_act, prev_room))

            if prev_act.lower() in self.ABBRV_DICT.keys():
                prev_act = defines.ABBRV_DICT[prev_act.lower()]
                add_triples.add((self.room, prev_act, prev_room))

            surr_objs = entities['object_surr']
            for s in surr_objs:
                add_triples.add((s, 'in', self.room))
                if self.graph_state.has_edge('you', s):
                    remove_triples.add(('you', 'have', s))

            inv_objs = entities['objs_inv']
            for i in inv_objs:
                add_triples.add(('you', 'have', i))
                if self.graph_state.has_edge(i, self.room):
                    remove_triples.add((i, 'in', self.room))

            attributes = entities['attributes']
            for o in inv_objs + surr_objs:
                if o in attributes.keys():
                    a_curr = attributes[o]
                    for a in a_curr:
                        add_triples.add((o, 'is', a))

        for rule in add_triples:
            u = '_'.join(str(rule[0]).split()).lower()
            v = '_'.join(str(rule[2]).split()).lower()
            r = '_'.join(str(rule[1]).split()).lower()
            if u in self.vocab_kge.keys() and v in self.vocab_kge.keys() and r in self.vocab_kgr.keys():
                self.graph_state.add_edge(u, v, rel=r)
        for rule in remove_triples:
            u = '_'.join(str(rule[0]).split()).lower()
            v = '_'.join(str(rule[2]).split()).lower()
            if u in self.vocab_kge.keys() and v in self.vocab_kge.keys():
                if self.graph_state.has_edge(u, v):
                    # print("REMOVE", (u, v))
                    self.graph_state.remove_edge(u, v)
        # self.visualize()
        return add_triples, entities

    def get_state_rep_kge(self):
        ret_ent = []
        ret_rel = []
        self.adj_matrix = np.array([np.zeros((len(self.vocab_kge), len(self.vocab_kge)))] * len(self.vocab_kgr))

        for u, v in self.graph_state.edges:
            r = self.graph_state.edges[u, v]['rel']
            r = '_'.join(str(r).split())
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.vocab_kge.keys() or v not in self.vocab_kge.keys():
                break

            u_idx = self.vocab_kge[u]
            v_idx = self.vocab_kge[v]
            r_idx = self.vocab_kgr[r]
            # print(u, v)
            self.adj_matrix[r_idx][u_idx][v_idx] = 1

            ret_ent.append(self.vocab_kge[u])
            ret_ent.append(self.vocab_kge[v])
            ret_rel.append(self.vocab_kgr[r])

        return (list(set(ret_ent)), list(set(ret_rel)))

    def get_obs_rep(self, *args):
        ret = [self.get_visible_state_rep_drqa(ob) for ob in args]
        return pad_sequences(ret, maxlen=300)

    def get_visible_state_rep_drqa(self, state_description):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']

        for rm in remove:
            state_description = state_description.replace(rm, '')

        return self.sp.encode_as_ids(state_description)

    def get_action_rep_drqa(self, action):

        action_desc_num = 20 * [0]
        action = str(action)

        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_len]
            action_desc_num[i] = self.vocab_act_rev[short_tok] if short_tok in self.vocab_act_rev else 0

        return action_desc_num

    def step(self, visible_state, inventory_state, objs, prev_action=None, cache=None, gat=True):
        ret, ret_cache = self.update_state(visible_state, inventory_state, objs, prev_action, cache)

        self.pruned_actions_rep = [self.get_action_rep_drqa(a) for a in self.vis_pruned_actions]

        inter = self.visible_state #+ "The actions are:" + ",".join(self.vis_pruned_actions) + "."
        self.drqa_input = self.get_visible_state_rep_drqa(inter)

        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix

        return ret, ret_cache


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x
