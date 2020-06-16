import collections
import redis
import numpy as np
from representations import StateAction
import random
import jericho
from jericho.template_action_generator import *
from jericho.defines import TemplateAction

GraphInfo = collections.namedtuple('GraphInfo', 'objs, ob_rep, act_rep, graph_state, graph_state_rep, admissible_actions, admissible_actions_rep')


def load_vocab(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: i for i, v in vocab.items()}
    return vocab, vocab_rev

def clean_obs(s):
    garbage_chars = ['\n', '*', '-', '(', ')', '<', '>']
    for c in garbage_chars:
        s = s.replace(c, ' ')
    return s.strip()


class QBERTEnv:
    '''

    QBERT environment performs additional graph-based processing.

    '''
    def __init__(self, rom_path, seed, spm_model, tsv_file, attr_file, step_limit=None, stuck_steps=10, gat=True, askbert=False, clear_kg=False):
        random.seed(seed)
        np.random.seed(seed)
        self.rom_path        = rom_path
        self.seed            = seed
        self.episode_steps   = 0
        self.stuck_steps     = 0
        self.valid_steps     = 0
        self.spm_model       = spm_model
        self.tsv_file        = tsv_file
        self.attr_file       = attr_file
        self.step_limit      = step_limit
        self.max_stuck_steps = stuck_steps
        self.gat             = gat
        self.env             = None
        self.conn_valid      = None
        self.conn_openie     = None
        self.vocab           = None
        self.vocab_rev       = None
        self.state_rep       = None
        self.askbert         = askbert
        self.clear_kg        = clear_kg

    def create(self):
        ''' Create the Jericho environment and connect to redis. '''
        self.env = jericho.FrotzEnv(self.rom_path, self.seed)
        self.bindings = jericho.load_bindings(self.rom_path)
        self.act_gen = TemplateActionGenerator(self.bindings)
        self.max_word_len = self.bindings['max_word_length']
        self.vocab, self.vocab_rev = load_vocab(self.env)
        self.conn_valid = redis.Redis(host='localhost', port=6379, db=0)
        self.conn_openie = redis.Redis(host='localhost', port=6379, db=1)
        self.conn_openie.flushdb()

    def _get_admissible_actions(self, objs):
        ''' Queries Redis for a list of admissible actions from the current state. '''
        obj_ids = [self.vocab_rev[o[:self.max_word_len]] for o in objs]
        world_state_hash = self.env.get_world_state_hash()
        admissible = self.conn_valid.get(world_state_hash)
        if admissible is None:
            possible_acts = self.act_gen.generate_template_actions(objs, obj_ids)
            admissible = self.env.find_valid_actions(possible_acts)
            redis_valid_value = '/'.join([str(a) for a in admissible])
            self.conn_valid.set(world_state_hash, redis_valid_value)
        else:
            try:
                admissible = [eval(a.strip()) for a in admissible.decode('cp1252').split('/')]
            except Exception as e:
                print("Exception: {}. Admissible: {}".format(e, admissible))
        return admissible

    def _build_graph_rep(self, action, ob_r):
        ''' Returns various graph-based representations of the current state. '''
        objs = [o[0] for o in self.env.identify_interactive_objects(ob_r)]
        objs.append('all')
        admissible_actions = self._get_admissible_actions(objs)
        admissible_actions_rep = [self.state_rep.get_action_rep_drqa(a.action) \
                                  for a in admissible_actions] \
                                      if admissible_actions else [[0] * 20]
        try: # Gather additional information about the new state
            get_state = self.env.get_state()
            ob_l = self.env.step('look')[0]
            self.env.set_state(get_state)
            ob_i = self.env.step('inventory')[0]
            self.env.set_state(get_state)
        except RuntimeError:
            print('RuntimeError: {}'.format(clean_obs(ob_r)))
            ob_l = ob_i = ''
        ob_rep = self.state_rep.get_obs_rep(ob_l, ob_i, ob_r, action)
        cleaned_obs = clean_obs('[loc] ' + ob_l + ' [inv] ' + ob_i + ' [obs] ' + ob_r)
        openie_cache = None
        openie_cache = self.conn_openie.get(cleaned_obs)
        if openie_cache is None:
            rules, tocache = self.state_rep.step(cleaned_obs, ob_i, objs, action, cache=None, gat=self.gat)
            if tocache is not None:
                self.conn_openie.set(cleaned_obs, str(tocache))
        else:
            openie_cache = eval(openie_cache.decode('cp1252'))
            rules, _ = self.state_rep.step(cleaned_obs, ob_i, objs, action, cache=openie_cache, gat=self.gat)
        graph_state = self.state_rep.graph_state
        graph_state_rep = self.state_rep.graph_state_rep
        action_rep = self.state_rep.get_action_rep_drqa(action)
        return GraphInfo(objs, ob_rep, action_rep, graph_state, graph_state_rep,\
                         admissible_actions, admissible_actions_rep), ob_l

    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        info['valid'] = self.env.world_changed() or done
        info['steps'] = self.episode_steps
        if info['valid']:
            self.valid_steps += 1
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1
        if (self.step_limit and self.valid_steps >= self.step_limit) \
           or self.stuck_steps > self.max_stuck_steps:
            done = True
        if done:
            graph_info, info['look'] = GraphInfo(objs=['all'],
                                   ob_rep=self.state_rep.get_obs_rep(obs, obs, obs, action),
                                   act_rep=self.state_rep.get_action_rep_drqa(action),
                                   graph_state=self.state_rep.graph_state,
                                   graph_state_rep=self.state_rep.graph_state_rep,
                                   admissible_actions=[],
                                   admissible_actions_rep=[]), 'done'
        else:
            try:
                graph_info, info['look'] = self._build_graph_rep(action, obs)
            except:
                graph_info, info['look'] = GraphInfo(objs=['all'],
                                   ob_rep=self.state_rep.get_obs_rep(obs, obs, obs, action),
                                   act_rep=self.state_rep.get_action_rep_drqa(action),
                                   graph_state=self.state_rep.graph_state,
                                   graph_state_rep=self.state_rep.graph_state_rep,
                                   admissible_actions=[],
                                   admissible_actions_rep=[]), 'done'
                done = True
        return obs, reward, done, info, graph_info

    def reset(self):
        self.state_rep = StateAction(self.spm_model, self.vocab, self.vocab_rev,
                                     self.tsv_file, self.max_word_len, self.askbert, self.attr_file)
        self.stuck_steps = 0
        self.valid_steps = 0
        self.episode_steps = 0
        obs, info = self.env.reset()
        info['valid'] = False
        info['steps'] = 0
        graph_info, info['look'] = self._build_graph_rep('look', obs)
        return obs, info, graph_info

    def soft_reset(self, gs, obs):
        self.stuck_steps = 0
        self.valid_steps = 0
        self.episode_steps = 0
        if self.clear_kg or gs is None:
            self.state_rep = StateAction(self.spm_model, self.vocab, self.vocab_rev,
                                         self.tsv_file, self.max_word_len, self.askbert, self.attr_file)
        else:
            graph_state, graph_state_rep = gs
            self.state_rep.graph_state = graph_state
            self.state_rep.graph_state_rep = graph_state_rep
        graph_info, _ = self._build_graph_rep('look', obs)
        return graph_info
    
    def clear_kgs(self):
        self.state_rep = StateAction(self.spm_model, self.vocab, self.vocab_rev,
                                     self.tsv_file, self.max_word_len, self.askbert, self.attr_file)

    def close(self):
        self.env.close()
