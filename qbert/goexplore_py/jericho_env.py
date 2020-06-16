from .import_ai import *
import collections
import redis
import numpy as np
from representations import StateAction
import random
import jericho
from jericho.template_action_generator import *
from jericho.defines import TemplateAction
from intrinsic_qbert import QBERTTrainer
import sys
np.set_printoptions(threshold=sys.maxsize)

GLOBAL_MAX_SCORE = -1

class ZorkPos:
    __slots__ = ['env_str', 'qbert', 'tuple']

    def __init__(self, env_str, qbert=None):
        self.env_str = env_str
        self.qbert = qbert

        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.env_str,self.qbert)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, ZorkPos):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        # self.env_str = d
        self.tuple = d

    def __repr__(self):
        return f'env_str={self.env_str}'


GraphInfo = collections.namedtuple('GraphInfo', 'objs, ob_rep, act_rep, graph_state, graph_state_rep, admissible_actions, admissible_actions_rep')

def load_vocab(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: i for i, v in vocab.items()}
    return vocab, vocab_rev

def clean_obs(s):
    garbage_chars = ['*', '-', '!', '[', ']']
    for c in garbage_chars:
        s = s.replace(c, ' ')
    return s.strip()


class MyZork:
    def __init__(self, params):
        # params = parse_args()
        # print(params)
        self.trainer = QBERTTrainer(params)
        # trainer.train(params['steps'])
        self.obs = None
        self.infos = None
        self.graph_infos = None
        self.IM = [set()] 
        # self.logger = 'logs/goexplore.log'
        self.logger = params['goexplore_logger']
        self.episode_steps = 0
    def write_log(self, log_str):
        with open(self.logger, 'a+') as fh:
            fh.write(log_str)

    def reset(self):
        # self.state_rep = StateAction(self.spm_model, self.vocab, self.vocab_rev,
        #                              self.tsv_file, self.max_word_len)
        # self.stuck_steps = 0
        # self.valid_steps = 0
        # self.episode_steps = 0
        # obs, info = self.env.reset()
        # info['valid'] = False
        # info['steps'] = 0
        # graph_info = self._build_graph_rep('look', obs)
        self.trainer.vec_env.go_reset()
        # return copy.copy(graph_info)
        #return copy.copy(obs), info, graph_info
        #return obs, info, graph_info

    def get_restore(self):
        get_state = self.trainer.vec_env.get_env_str()[0]
        score = self.trainer.vec_env.get_score()[0]
        moves = self.trainer.vec_env.get_moves()[0]
        return  (
            get_state,
            score,
            moves,
            self.trainer.model.state_dict(),
            self.obs,
            self.infos,
            self.graph_infos,
            self.IM

        )

    def restore(self, data):
        #TODO: implement
        # (full_state, state, score, steps, pos, room_time, ram_death_state, self.score_objects, self.cur_lives) = data
        # self.state = copy.copy(state)
        # self.env.reset()
        # self.unwrapped.restore_full_state(full_state)
        # self.ram = self.env.unwrapped.ale.getRAM()
        # self.cur_score = score
        # self.cur_steps = steps
        # self.pos = pos
        # self.room_time = room_time
        # self.ram_death_state = ram_death_state
        # return copy.copy(self.state)
        get_state, score, steps, qbert_state, obs, infos, graph_infos, IM = data
        self.cur_score = score
        self.cur_steps = steps
        self.trainer.vec_env.go_reset()
        # self.env.reset()
        # self.env.set_state(get_state)
        self.trainer.vec_env.go_load_from(get_state)
        self.trainer.model.load_state_dict(qbert_state)
        self.obs = obs
        self.infos = infos
        self.graph_infos = graph_infos
        self.IM = IM
        cur_score = self.trainer.vec_env.get_score()[0]
        print ("restoring cell: score:{} steps:{}".format(cur_score, self.cur_steps))
        #print ("restored, with score: " + str(self.env.get_score()))
        #return copy.copy(self.state)


    def step(self, max_steps=1):
        global GLOBAL_MAX_SCORE
        self.episode_steps += 1
        # obs, reward, done, info = self.env.step(action)
        # info['valid'] = self.env.world_changed() or done
        # info['steps'] = self.episode_steps
        # if info['valid']:
        #     self.valid_steps += 1
        #     self.stuck_steps = 0
        # else:
        #     self.stuck_steps += 1
        # if (self.step_limit and self.valid_steps >= self.step_limit) \
        #    or self.stuck_steps > self.max_stuck_steps:
        #     done = True
        # if done:
        #     graph_info = GraphInfo(objs=['all'],
        #                            ob_rep=self.state_rep.get_obs_rep(obs, obs, obs, action),
        #                            act_rep=self.state_rep.get_action_rep_drqa(action),
        #                            graph_state=self.state_rep.graph_state,
        #                            graph_state_rep=self.state_rep.graph_state_rep,
        #                            admissible_actions=[],
        #                            admissible_actions_rep=[])
        # else:
        #     graph_info = self._build_graph_rep(action, obs)
        obs, rewards, dones, infos, graph_infos, scores, chosen_actions, IM = self.trainer.goexplore_train(self.obs,
            self.infos, self.graph_infos, max_steps=max_steps, INTRINSIC_MOTIVTATION=self.IM)


        cur_score = self.trainer.vec_env.get_score()[0]
        self.write_log("explored:{}, score:{},{},max:{}\n".format(self.episode_steps, cur_score, infos, GLOBAL_MAX_SCORE))
        if cur_score > GLOBAL_MAX_SCORE:
            tqdm.write(f'NEW MAX FOUND: {cur_score}')
            GLOBAL_MAX_SCORE = cur_score
            print (infos)
        self.obs = obs
        self.infos = infos
        self.graph_infos = graph_infos
        self.IM = IM
        return obs, rewards, dones, infos, graph_infos, scores, chosen_actions, IM
        #return copy.copy(graph_info), reward, done, info
        #return copy.copy(obs), reward, done, info, graph_info

    def get_pos(self):
        #print (self.env.get_state())
        get_state = self.trainer.vec_env.get_env_str()[0]
        return ZorkPos(str(get_state))

