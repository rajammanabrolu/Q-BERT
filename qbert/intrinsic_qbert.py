import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import os
from os.path import basename, splitext
import numpy as np
import time
import sentencepiece as spm
from statistics import mean

from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import unabbreviate, clean
import jericho.defines

# from representations import StateAction
from models import QBERT
from env import *
from vec_env import *
import logger

import random
# from extraction import kgextraction
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
device = torch.device("cuda")


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    logger.set_level(60)
    log = logger.log


class QBERTTrainer(object):
    '''

    QBERT main class.


    '''
    def __init__(self, params):
        torch.manual_seed(params['seed'])
        np.random.seed(params['seed'])
        random.seed(params['seed'])
        configure_logger(params['output_dir'])
        log('Parameters {}'.format(params))
        self.params = params
        self.chkpt_path = os.path.dirname(self.params['checkpoint_path'])
        if not os.path.exists(self.chkpt_path):
            os.mkdir(self.chkpt_path)
        self.binding = load_bindings(params['rom_file_path'])
        self.max_word_length = self.binding['max_word_length']
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(params['spm_file'])
        #askbert_args = {'input_text': '', 'length': 10, 'batch_size': 1, 'temperature': 1, 'model_name': '117M',
        #                'seed': 0, 'nsamples': 10, 'cutoffs': "6.5 -7 -5", 'write_sfdp': False, 'random': False}
        #self.extraction = kgextraction.World([], [], [], askbert_args)
        self.askbert = params['extraction']
        kg_env = QBERTEnv(params['rom_file_path'], params['seed'], self.sp,
                          params['tsv_file'], params['attr_file'], step_limit=params['reset_steps'],
                          stuck_steps=params['stuck_steps'], gat=params['gat'], askbert=self.askbert,
                          clear_kg=params['clear_kg_on_reset'])

        self.vec_env = VecEnv(params['batch_size'], kg_env, params['openie_path'], params['redis_db_path'],
                              params['buffer_size'],params['extraction'], params['training_type'],
                              params['clear_kg_on_reset'])
        self.template_generator = TemplateActionGenerator(self.binding)
        env = FrotzEnv(params['rom_file_path'])
        self.max_game_score = env.get_max_score()
        self.cur_reload_state = env.get_state()
        self.vocab_act, self.vocab_act_rev = load_vocab(env)
        self.model = QBERT(params, self.template_generator.templates, self.max_word_length,
                           self.vocab_act, self.vocab_act_rev, len(self.sp), gat=self.params['gat']).cuda()
        self.batch_size = params['batch_size']
        if params['preload_weights']:
            self.model = torch.load(self.params['preload_weights'])['model']
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])

        self.loss_fn1 = nn.BCELoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss()
        self.loss_fn3 = nn.MSELoss()

        self.chained_logger = params['chained_logger']
        self.total_steps = 0
    
    def log_file(self, str):
        with open(self.chained_logger, 'a+') as fh:
            fh.write(str)


    def generate_targets(self, admissible, objs):
        '''
        Generates ground-truth targets for admissible actions.

        :param admissible: List-of-lists of admissible actions. Batch_size x Admissible
        :param objs: List-of-lists of interactive objects. Batch_size x Objs
        :returns: template targets and object target tensors

        '''
        tmpl_target = []
        obj_targets = []
        for adm in admissible:
            obj_t = set()
            cur_t = [0] * len(self.template_generator.templates)
            for a in adm:
                cur_t[a.template_id] = 1
                obj_t.update(a.obj_ids)
            tmpl_target.append(cur_t)
            obj_targets.append(list(obj_t))
        tmpl_target_tt = torch.FloatTensor(tmpl_target).cuda()

        # Note: Adjusted to use the objects in the admissible actions only
        object_mask_target = []
        for objl in obj_targets: # in objs
            cur_objt = [0] * len(self.vocab_act)
            for o in objl:
                cur_objt[o] = 1
            object_mask_target.append([[cur_objt], [cur_objt]])
        obj_target_tt = torch.FloatTensor(object_mask_target).squeeze().cuda()
        return tmpl_target_tt, obj_target_tt


    def generate_graph_mask(self, graph_infos):
        assert len(graph_infos) == self.batch_size
        mask_all = []
        # TODO use graph dropout for masking here
        for graph_info in graph_infos:
            mask = [0] * len(self.vocab_act.keys())
            if self.params['masking'] == 'kg':
                # Uses the knowledge graph as the mask.
                graph_state = graph_info.graph_state
                ents = set()
                for u, v in graph_state.edges:
                    ents.add(u)
                    ents.add(v)
                for ent in ents:
                    for ent_word in ent.split():
                        if ent_word[:self.max_word_length] in self.vocab_act_rev:
                            idx = self.vocab_act_rev[ent_word[:self.max_word_length]]
                            mask[idx] = 1
                if self.params['mask_dropout'] != 0:
                    drop = random.sample(range(0, len(self.vocab_act.keys()) - 1),
                                         int(self.params['mask_dropout'] * len(self.vocab_act.keys())))
                    for i in drop:
                        mask[i] = 1
            elif self.params['masking'] == 'interactive':
                # Uses interactive objects grount truth as the mask.
                for o in graph_info.objs:
                    o = o[:self.max_word_length]
                    if o in self.vocab_act_rev.keys() and o != '':
                        mask[self.vocab_act_rev[o]] = 1
                    if self.params['mask_dropout'] != 0:
                        drop = random.sample(range(0, len(self.vocab_act.keys()) - 1),
                                             int(self.params['mask_dropout'] * len(self.vocab_act.keys())))
                        for i in drop:
                            mask[i] = 1
            elif self.params['masking'] == 'none':
                # No mask at all.
                mask = [1] * len(self.vocab_act.keys())
            else:
                assert False, 'Unrecognized masking {}'.format(self.params['masking'])
            mask_all.append(mask)
        return torch.BoolTensor(mask_all).cuda().detach()


    def discount_reward(self, transitions, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(transitions))):
            _, _, values, rewards, done_masks, _, _, _, _, _, _ = transitions[t]
            R = rewards + self.params['gamma'] * R * done_masks
            adv = R - values
            returns.append(R)
            advantages.append(adv)
        return returns[::-1], advantages[::-1]

    def goexplore_train(self, obs, infos, graph_infos, max_steps, INTRINSIC_MOTIVTATION):
        start = time.time()
        transitions = []
        if obs == None:
            obs, infos, graph_infos = self.vec_env.go_reset()
        for step in range(1, max_steps + 1):
            self.total_steps += 1
            tb.logkv('Step', self.total_steps)
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]
            # scores = [info['score'] for info in infos]
            if self.params['reward_type'] == 'game_only':
                scores = [info['score'] for info in infos]
            elif self.params['reward_type'] == 'IM_only':
                scores = np.array([int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in range(self.params['batch_size'])])
            elif self.params['reward_type'] == 'game_and_IM':
                scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * ((infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in range(self.params['batch_size'])])
            
            tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
                obs_reps, scores, graph_state_reps, graph_mask_tt)
            tb.logkv_mean('Value', value.mean().item())

            # Log some of the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()]
            tmpl_pred_str = ', '.join(['{} {:.3f}'.format(tmpl, prob) for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())])

            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)

            gt_tmpls = [self.template_generator.templates[i] for i in tmpl_gt_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            gt_objs = [self.vocab_act[i] for i in obj_mask_gt_tt[0,0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            log('TmplPred: {} GT: {}'.format(tmpl_pred_str, ', '.join(gt_tmpls)))
            topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0,0]).topk(5)
            topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
            o1_pred_str = ', '.join(['{} {:.3f}'.format(o, prob) for o, prob in zip(topk_o1, topk_o1_probs.tolist())])
            graph_mask_str = [self.vocab_act[i] for i in graph_mask_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            log('ObjtPred: {} GT: {} Mask: {}'.format(o1_pred_str, ', '.join(gt_objs), ', '.join(graph_mask_str)))

            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)

            # Chooses random valid-actions to execute

            obs, rewards, dones, infos, graph_infos = self.vec_env.go_step(chosen_actions)

            edges = [set(graph_info.graph_state.edges) for graph_info in graph_infos]
            size_updates = [0] * self.params['batch_size']
            for i, s in enumerate(INTRINSIC_MOTIVTATION):
                orig_size = len(s)
                s.update(edges[i])
                size_updates[i] = len(s) - orig_size
            rewards = list(rewards)
            for i in range(self.params['batch_size']):
                if self.params['reward_type'] == 'IM_only':
                    rewards[i] = size_updates[i] * self.params['intrinsic_motivation_factor']
                elif self.params['reward_type'] == 'game_and_IM':
                    rewards[i] += size_updates[i] * self.params['intrinsic_motivation_factor']
            rewards = tuple(rewards)

            tb.logkv_mean('TotalStepsPerEpisode', sum([i['steps'] for i in infos]) / float(len(graph_infos)))
            tb.logkv_mean('Valid', infos[0]['valid'])
            log('Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}'.format(
                chosen_actions[0], rewards[0], infos[0]['score'], dones[0], value[0].item()))
            log('Obs: {}'.format(clean(obs[0])))
            if dones[0]:
                log('Step {} EpisodeScore {}\n'.format(step, infos[0]['score']))
            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean('EpisodeScore', info['score'])
            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)

            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append((tmpl_pred_tt, obj_pred_tt, value, rew_tt,
                                done_mask_tt, tmpl_gt_tt, dec_tmpl_tt,
                                dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps))

            if len(transitions) >= self.params['bptt']:
                tb.logkv('StepsPerSecond', float(step) / (time.time() - start))
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                # scores = [info['score'] for info in infos]
                if self.params['reward_type'] == 'game_only':
                    scores = [info['score'] for info in infos]
                elif self.params['reward_type'] == 'IM_only':
                    scores = np.array([int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in range(self.params['batch_size'])])
                elif self.params['reward_type'] == 'game_and_IM':
                    scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * ((infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in range(self.params['batch_size'])])
                _, _, _, _, next_value, _ = self.model(obs_reps, scores, graph_state_reps, graph_mask_tt)
                returns, advantages = self.discount_reward(transitions, next_value)
                log('Returns: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in returns]))
                log('Advants: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in advantages]))
                tb.logkv_mean('Advantage', advantages[-1].median().item())
                loss = self.update(transitions, returns, advantages)
                del transitions[:]
                self.model.restore_hidden()

            if step % self.params['checkpoint_interval'] == 0:
                parameters = { 'model': self.model }
                torch.save(parameters, os.path.join(self.params['output_dir'], 'qbert.pt'))

        # self.vec_env.close_extras()
        return obs, rewards, dones, infos, graph_infos, scores, chosen_actions, INTRINSIC_MOTIVTATION


    def train(self, max_steps):
        start = time.time()
        if self.params['training_type'] == 'chained':
            self.log_file("BEGINNING OF TRAINING: patience={}, max_n_steps_back={}\n".format(self.params['patience'], self.params['buffer_size']))
        frozen_policies = []
        transitions = []
        self.back_step = -1

        previous_best_seen_score = float("-inf")
        previous_best_step = 0
        previous_best_state = None
        previous_best_snapshot = None
        previous_best_ACTUAL_score = 0
        self.cur_reload_step = 0
        force_reload = [False] * self.params['batch_size']
        last_edges = None

        self.valid_track = np.zeros(self.params['batch_size'])
        self.stagnant_steps = 0

        INTRINSIC_MOTIVTATION= [set() for i in range(self.params['batch_size'])]

        obs, infos, graph_infos, env_str = self.vec_env.reset()
        snap_obs = obs[0]
        snap_info = infos[0]
        snap_graph_reps = None
        # print (obs)
        # print (infos)
        # print (graph_infos)
        for step in range(1, max_steps + 1):
            wallclock = time.time()

            if any(force_reload) and self.params['training_type'] == 'chained':
                num_reload = force_reload.count(True)
                t_obs = np.array(obs)
                t_obs[force_reload] = [snap_obs] * num_reload
                obs = tuple(t_obs)

                t_infos = np.array(infos)
                t_infos[force_reload] = [snap_info] * num_reload
                infos = tuple(t_infos)

                t_graphs = list(graph_infos)
                # namedtuple gets lost in np.array
                t_updates = self.vec_env.load_from(self.cur_reload_state, force_reload, snap_graph_reps, snap_obs)
                for i in range(self.params['batch_size']):
                    if force_reload[i]:
                        t_graphs[i] = t_updates[i]
                graph_infos = tuple(t_graphs)

                force_reload = [False] * self.params['batch_size']


            tb.logkv('Step', step)
            obs_reps = np.array([g.ob_rep for g in graph_infos])
            graph_mask_tt = self.generate_graph_mask(graph_infos)
            graph_state_reps = [g.graph_state_rep for g in graph_infos]

            if self.params['reward_type'] == 'game_only':
                scores = [info['score'] for info in infos]
            elif self.params['reward_type'] == 'IM_only':
                scores = np.array([int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in range(self.params['batch_size'])])
            elif self.params['reward_type'] == 'game_and_IM':
                scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * ((infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in range(self.params['batch_size'])])
            tmpl_pred_tt, obj_pred_tt, dec_obj_tt, dec_tmpl_tt, value, dec_steps = self.model(
                obs_reps, scores, graph_state_reps, graph_mask_tt)
            tb.logkv_mean('Value', value.mean().item())

            # Log the predictions and ground truth values
            topk_tmpl_probs, topk_tmpl_idxs = F.softmax(tmpl_pred_tt[0]).topk(5)
            topk_tmpls = [self.template_generator.templates[t] for t in topk_tmpl_idxs.tolist()]
            tmpl_pred_str = ', '.join(['{} {:.3f}'.format(tmpl, prob) for tmpl, prob in zip(topk_tmpls, topk_tmpl_probs.tolist())])

            # Generate the ground truth and object mask
            admissible = [g.admissible_actions for g in graph_infos]
            objs = [g.objs for g in graph_infos]
            tmpl_gt_tt, obj_mask_gt_tt = self.generate_targets(admissible, objs)

            # Log template/object predictions/ground_truth
            gt_tmpls = [self.template_generator.templates[i] for i in tmpl_gt_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            gt_objs = [self.vocab_act[i] for i in obj_mask_gt_tt[0,0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            log('TmplPred: {} GT: {}'.format(tmpl_pred_str, ', '.join(gt_tmpls)))
            topk_o1_probs, topk_o1_idxs = F.softmax(obj_pred_tt[0,0]).topk(5)
            topk_o1 = [self.vocab_act[o] for o in topk_o1_idxs.tolist()]
            o1_pred_str = ', '.join(['{} {:.3f}'.format(o, prob) for o, prob in zip(topk_o1, topk_o1_probs.tolist())])
            # graph_mask_str = [self.vocab_act[i] for i in graph_mask_tt[0].nonzero().squeeze().cpu().numpy().flatten().tolist()]
            log('ObjtPred: {} GT: {}'.format(o1_pred_str, ', '.join(gt_objs))) # , ', '.join(graph_mask_str)))

            chosen_actions = self.decode_actions(dec_tmpl_tt, dec_obj_tt)

            #stepclock = time.time()

            obs, rewards, dones, infos, graph_infos, env_str = self.vec_env.step(chosen_actions)

            #print('stepclock', time.time() - stepclock)
            self.valid_track += [info['valid'] for info in infos]
            self.stagnant_steps += 1
            force_reload = list(dones)

            edges = [set(graph_info.graph_state.edges) for graph_info in graph_infos]
            size_updates = [0] * self.params['batch_size']
            for i, s in enumerate(INTRINSIC_MOTIVTATION):
                orig_size = len(s)
                s.update(edges[i])
                size_updates[i] = len(s) - orig_size
            rewards = list(rewards)
            for i in range(self.params['batch_size']):
                if self.params['reward_type'] == 'IM_only':
                    rewards[i] = size_updates[i] * self.params['intrinsic_motivation_factor']
                elif self.params['reward_type'] == 'game_and_IM':
                    rewards[i] += size_updates[i] * self.params['intrinsic_motivation_factor']
            rewards = tuple(rewards)

            if last_edges:
                stayed_same = [1 if (len(edges[i] - last_edges[i]) <= self.params['kg_diff_threshold']) else 0 for i in range(self.params['batch_size'])]
                # print ("stayed_same: {}".format(stayed_same))
            valid_kg_update = last_edges and sum(stayed_same) / self.params['batch_size'] > self.params['kg_diff_batch_percentage']
            last_edges = edges

            snapshot = self.vec_env.get_snapshot()
            real_scores = np.array([infos[i]['score'] for i in range(len(rewards))])

            if self.params['reward_type'] == 'game_only':
                scores = [info['score'] for info in infos]
            elif self.params['reward_type'] == 'IM_only':
                scores = np.array([int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in range(self.params['batch_size'])])
            elif self.params['reward_type'] == 'game_and_IM':
                scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * ((infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in range(self.params['batch_size'])])
            cur_max_score_idx = np.argmax(scores)
            if scores[cur_max_score_idx] > previous_best_seen_score and self.params['training_type'] == 'chained': # or valid_kg_update:
                print ("New Reward Founded OR KG updated")
                previous_best_step = step
                previous_best_state = env_str[cur_max_score_idx]
                previous_best_seen_score = scores[cur_max_score_idx]
                previous_best_snapshot = snapshot[cur_max_score_idx]
                self.back_step = -1
                self.valid_track = np.zeros(self.params['batch_size'])
                self.stagnant_steps = 0
                print ("\tepoch: {}".format(previous_best_step))
                print ("\tnew score: {}".format(previous_best_seen_score))
                print ("\tthis info: {}".format(infos[cur_max_score_idx]))
                self.log_file("New High Score Founded: step:{}, new_score:{}, infos:{}\n".format(previous_best_step, previous_best_seen_score, infos[cur_max_score_idx]))

            previous_best_ACTUAL_score = max(np.max(real_scores), previous_best_ACTUAL_score)
            print ("step {}: scores: {}, best_real_score: {}".format(step, scores, previous_best_ACTUAL_score))

            tb.logkv_mean('TotalStepsPerEpisode', sum([i['steps'] for i in infos]) / float(len(graph_infos)))
            tb.logkv_mean('Valid', infos[0]['valid'])
            log('Act: {}, Rew {}, Score {}, Done {}, Value {:.3f}'.format(
                chosen_actions[0], rewards[0], infos[0]['score'], dones[0], value[0].item()))
            log('Obs: {}'.format(clean(obs[0])))
            if dones[0]:
                log('Step {} EpisodeScore {}\n'.format(step, infos[0]['score']))
            for done, info in zip(dones, infos):
                if done:
                    tb.logkv_mean('EpisodeScore', info['score'])
            rew_tt = torch.FloatTensor(rewards).cuda().unsqueeze(1)
            done_mask_tt = (~torch.tensor(dones)).float().cuda().unsqueeze(1)
            self.model.reset_hidden(done_mask_tt)
            transitions.append((tmpl_pred_tt, obj_pred_tt, value, rew_tt,
                                done_mask_tt, tmpl_gt_tt, dec_tmpl_tt,
                                dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps))

            if len(transitions) >= self.params['bptt']:
                tb.logkv('StepsPerSecond', float(step) / (time.time() - start))
                self.model.clone_hidden()
                obs_reps = np.array([g.ob_rep for g in graph_infos])
                graph_mask_tt = self.generate_graph_mask(graph_infos)
                graph_state_reps = [g.graph_state_rep for g in graph_infos]
                if self.params['reward_type'] == 'game_only':
                    scores = [info['score'] for info in infos]
                elif self.params['reward_type'] == 'IM_only':
                    scores = np.array([int(len(INTRINSIC_MOTIVTATION[i]) * self.params['intrinsic_motivation_factor']) for i in range(self.params['batch_size'])])
                elif self.params['reward_type'] == 'game_and_IM':
                    scores = np.array([infos[i]['score'] + (len(INTRINSIC_MOTIVTATION[i]) * ((infos[i]['score'] + self.params['epsilon']) / self.max_game_score)) for i in range(self.params['batch_size'])])
                _, _, _, _, next_value, _ = self.model(obs_reps, scores, graph_state_reps, graph_mask_tt)
                returns, advantages = self.discount_reward(transitions, next_value)
                log('Returns: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in returns]))
                log('Advants: ', ', '.join(['{:.3f}'.format(a[0].item()) for a in advantages]))
                tb.logkv_mean('Advantage', advantages[-1].median().item())
                loss = self.update(transitions, returns, advantages)
                del transitions[:]
                self.model.restore_hidden()

            if step % self.params['checkpoint_interval'] == 0:
                parameters = { 'model': self.model }
                torch.save(parameters, os.path.join(self.params['output_dir'], 'qbert.pt'))

            bottleneck = self.params['training_type'] == 'chained' and \
                ((self.stagnant_steps >= self.params['patience'] and not self.params['patience_valid_only']) or 
                (self.params['patience_valid_only'] and sum(self.valid_track >= self.params['patience']) >= self.params['batch_size'] * self.params['patience_batch_factor']))
            if bottleneck:
                print ("Bottleneck detected at step: {}".format(step))
                # new_backstep += 1
                # new_back_step = (step - previous_best_step - self.params['patience']) // self.params['patience']
                self.back_step += 1
                if self.back_step == 0:
                    self.vec_env.import_snapshot(previous_best_snapshot)
                    cur_time = time.strftime("%Y%m%d-%H%M%S")
                    torch.save(self.model.state_dict(), os.path.join(self.chkpt_path, '{}.pt'.format(cur_time)))
                    frozen_policies.append((cur_time, previous_best_state))
                    # INTRINSIC_MOTIVTATION= [set() for i in range(self.params['batch_size'])]
                    self.log_file("Current model saved at: model/checkpoints/{}.pt\n".format(cur_time))
                self.model = QBERT(self.params, self.template_generator.templates, self.max_word_length,
                                   self.vocab_act, self.vocab_act_rev, len(self.sp), gat=self.params['gat']).cuda()


                if self.back_step >= self.params['buffer_size']:
                    print ("Buffer exhausted. Finishing training")
                    self.vec_env.close_extras()
                    return
                print (previous_best_snapshot[-1 - self.back_step])
                snap_obs, snap_info, snap_graph_reps, self.cur_reload_state = previous_best_snapshot[-1 - self.back_step]
                print ("Loading snapshot, infos: {}".format(snap_info))
                self.log_file("Loading snapshot, infos: {}\n".format(snap_info))
                self.cur_reload_step = previous_best_step
                force_reload = [True] * self.params['batch_size']
                self.valid_track = np.zeros(self.params['batch_size'])
                self.stagnant_steps = 0

                # print out observations here
                print ("Current observations: {}".format([info['look'] for info in infos]))
                print ("Previous_best_step: {}, step_back: {}".format(previous_best_step, self.back_step))
                self.log_file("Bottleneck Detected: step:{}, previous_best_step:{}, cur_step_back:{}\n".format(i, previous_best_step, self.back_step))
                self.log_file("Current observations: {}\n".format([info['look'] for info in infos]))
            #exit()

            


        self.vec_env.close_extras()

    


    def update(self, transitions, returns, advantages):
        assert len(transitions) == len(returns) == len(advantages)
        loss = 0
        for trans, ret, adv in zip(transitions, returns, advantages):
            tmpl_pred_tt, obj_pred_tt, value, _, _, tmpl_gt_tt, dec_tmpl_tt, \
                dec_obj_tt, obj_mask_gt_tt, graph_mask_tt, dec_steps = trans

            # Supervised Template Loss
            tmpl_probs = F.softmax(tmpl_pred_tt, dim=1)
            template_loss = self.params['template_coeff'] * self.loss_fn1(tmpl_probs, tmpl_gt_tt)

            # Supervised Object Loss
            if self.params['batch_size'] == 1:
                object_mask_target = obj_mask_gt_tt.unsqueeze(0).permute((1, 0, 2))
            else:
                object_mask_target = obj_mask_gt_tt.permute((1, 0, 2))
            obj_probs = F.softmax(obj_pred_tt, dim=2)
            object_mask_loss = self.params['object_coeff'] * self.loss_fn1(obj_probs, object_mask_target)

            # Build the object mask
            o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
            for d, st in enumerate(dec_steps):
                if st > 1:
                    o1_mask[d] = 1
                    o2_mask[d] = 1
                elif st == 1:
                    o1_mask[d] = 1
            o1_mask = torch.FloatTensor(o1_mask).cuda()
            o2_mask = torch.FloatTensor(o2_mask).cuda()

            # Policy Gradient Loss
            policy_obj_loss = torch.FloatTensor([0]).cuda()
            cnt = 0
            for i in range(self.batch_size):
                if dec_steps[i] >= 1:
                    cnt += 1
                    batch_pred = obj_pred_tt[0, i, graph_mask_tt[i]]
                    action_log_probs_obj = F.log_softmax(batch_pred, dim=0)
                    dec_obj_idx = dec_obj_tt[0,i].item()
                    graph_mask_list = graph_mask_tt[i].nonzero().squeeze().cpu().numpy().flatten().tolist()
                    idx = graph_mask_list.index(dec_obj_idx)
                    log_prob_obj = action_log_probs_obj[idx]
                    policy_obj_loss += -log_prob_obj * adv[i].detach()
            if cnt > 0:
                policy_obj_loss /= cnt
            tb.logkv_mean('PolicyObjLoss', policy_obj_loss.item())
            log_probs_obj = F.log_softmax(obj_pred_tt, dim=2)

            log_probs_tmpl = F.log_softmax(tmpl_pred_tt, dim=1)
            action_log_probs_tmpl = log_probs_tmpl.gather(1, dec_tmpl_tt).squeeze()

            policy_tmpl_loss = (-action_log_probs_tmpl * adv.detach().squeeze()).mean()
            tb.logkv_mean('PolicyTemplateLoss', policy_tmpl_loss.item())

            policy_loss = policy_tmpl_loss + policy_obj_loss

            value_loss = self.params['value_coeff'] * self.loss_fn3(value, ret)
            tmpl_entropy = -(tmpl_probs * log_probs_tmpl).mean()
            tb.logkv_mean('TemplateEntropy', tmpl_entropy.item())
            object_entropy = -(obj_probs * log_probs_obj).mean()
            tb.logkv_mean('ObjectEntropy', object_entropy.item())
            # Minimizing entropy loss will lead to increased entropy
            entropy_loss = self.params['entropy_coeff'] * -(tmpl_entropy + object_entropy)

            loss += template_loss + object_mask_loss + value_loss + entropy_loss + policy_loss

        tb.logkv('Loss', loss.item())
        tb.logkv('TemplateLoss', template_loss.item())
        tb.logkv('ObjectLoss', object_mask_loss.item())
        tb.logkv('PolicyLoss', policy_loss.item())
        tb.logkv('ValueLoss', value_loss.item())
        tb.logkv('EntropyLoss', entropy_loss.item())
        tb.dumpkvs()
        loss.backward()

        # Compute the gradient norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv('UnclippedGradNorm', grad_norm)

        nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])

        # Clipped Grad norm
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            grad_norm += p.grad.data.norm(2).item()
        tb.logkv('ClippedGradNorm', grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss


    def decode_actions(self, decoded_templates, decoded_objects):
        '''
        Returns string representations of the given template actions.

        :param decoded_template: Tensor of template indices.
        :type decoded_template: Torch tensor of size (Batch_size x 1).
        :param decoded_objects: Tensor of o1, o2 object indices.
        :type decoded_objects: Torch tensor of size (2 x Batch_size x 1).

        '''
        decoded_actions = []
        for i in range(self.batch_size):
            decoded_template = decoded_templates[i].item()
            decoded_object1 = decoded_objects[0][i].item()
            decoded_object2 = decoded_objects[1][i].item()
            decoded_action = self.tmpl_to_str(decoded_template, decoded_object1, decoded_object2)
            decoded_actions.append(decoded_action)
        return decoded_actions


    def tmpl_to_str(self, template_idx, o1_id, o2_id):
        """ Returns a string representation of a template action. """
        template_str = self.template_generator.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', self.vocab_act[o1_id])
        else:
            return template_str.replace('OBJ', self.vocab_act[o1_id], 1)\
                               .replace('OBJ', self.vocab_act[o2_id], 1)
