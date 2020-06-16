from .import_ai import *
from jericho.defines import TemplateAction
class ZorkPos:
    __slots__ = ['env_str', 'kg', 'tuple']

    def __init__(self, env_str, kg=None):
        self.env_str = env_str
        self.kg = kg

        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.env_str,self.kg)

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
@dataclass
class PosInfo:
    __slots__ = ['exact', 'cell', 'state', 'restore']
    exact: tuple
    cell: tuple
    state: typing.Any
    restore: typing.Any

@dataclass
class TrajectoryElement:
    __slots__ = ['from_', 'to', 'action', 'reward', 'done', 'real_pos']
    from_: PosInfo
    to: PosInfo
    action: int
    reward: float
    done: bool
    qbert: typing.Any
    real_pos: ZorkPos

class RandomExplorer:
    def init_seed(self):
        pass

    def init_trajectory(self, arg, arg2):
        pass

    def seen_state(self, e):
        pass

    def get_action(self, state, env):
        return random.randint(0, env.action_space.n - 1)

    def __repr__(self):
        return 'RandomExplorer()'


class RepeatedRandomExplorer:
    def __init__(self, mean_repeat=10):
        self.mean_repeat = mean_repeat
        self.action = 0
        self.remaining = 0

    def init_seed(self):
        self.remaining = 0

    def init_trajectory(self, arg, arg2):
        pass

    def seen_state(self, e):
        pass

    def get_action(self, state, env):
        if self.remaining <= 0:
            self.action = random.randint(0, env.action_space.n - 1)
            # Note, this is equivalent to selecting an action and then repeating it
            # with some probability.
            self.remaining = np.random.geometric(1 / self.mean_repeat)
        self.remaining -= 1
        return self.action

    def __repr__(self):
        return f'repeat-{self.mean_repeat}'


class RepeatedRandomExplorerRobot:
    def __init__(self, mean_repeat=10):
        self.mean_repeat = mean_repeat
        self.action = 0
        self.remaining = 0

    def init_seed(self):
        self.remaining = 0

    def init_trajectory(self, arg, arg2):
        pass

    def seen_state(self, e):
        pass

    def get_action(self, state, env):
        if self.remaining <= 0:
            self.action = env.action_space.sample()
            # Note, this is equivalent to selecting an action and then repeating it
            # with some probability.
            self.remaining = np.random.geometric(1 / self.mean_repeat)
        self.remaining -= 1
        return self.action

    def __repr__(self):
        return f'repeat-{self.mean_repeat}'


class DoNothingExplorer:
    def init_seed(self):
        pass

    def init_trajectory(self, arg, arg2):
        pass

    def seen_state(self, e):
        pass

    def get_action(self, *args):
        return ''

# class QBERTExplorer_Boltzmann:
#     # q_net and target_net
#     # snapshot one network and use that to explore,
#     # you make updates to another network the entire time you're exploring
#     # everytime you refresh you sync the weights of the two nets
#     #Boltzmann sample from the current policy
#     def __init__(self, mean_repeat=10):
#         self.mean_repeat = mean_repeat
#         self.action = 0
#         self.remaining = 0

#     def init_seed(self):
#         self.remaining = 0

#     def init_trajectory(self, arg, arg2):
#         pass

#     def seen_state(self, e):
#         pass

#     def get_action(self, state, env):
#         if self.remaining <= 0:
#             self.action = env.action_space.sample()
#             # Note, this is equivalent to selecting an action and then repeating it
#             # with some probability.
#             self.remaining = np.random.geometric(1 / self.mean_repeat)
#         self.remaining -= 1
#         return self.action

#     def __repr__(self):
#         return 'QBERTExplorer_Boltzmann()'

class QBERTExplorer:
    # vanilla snapshotting both QBERT weights and KG, restoring both
    def __init__(self):
        pass

    def init_seed(self):
        # self.target_model = None
        pass

    # def init_trajectory(self, arg, arg2):
    #     trainer = QBERTExplorer(params)
    #     return TrajectoryElement(
    #                 initial_pos_info,
    #                 self.get_pos_info(),
    #                 action, reward, done,
    #                 self.get_real_cell(),
    #                 trainer
    #             )
        # pass
    def init_trajectory(self, arg, arg2):
        pass

    def seen_state(self, e):
        pass

    def set_target_model(self, model):
        self.target_model = copy.deepcopy(model)

    # def get_trajectory(self, trainer, state, model):
    #     trainer.chosen_state = state
    #     trainer.model = model
    #     trainer.train()
    #     trajectory = trainer.last_trajectory
    #     return trajectory
    # def get_action(self, state, env):
    #     return env.step

    def __repr__(self):
        return 'QBERTExplorer()'

class GetActionRandomExplorer:
    def __init__(self, mean_repeat=10):
        self.mean_repeat = mean_repeat
        self.action = 0
        self.remaining = 0

    def init_seed(self):
        self.remaining = 0

    def init_trajectory(self, arg, arg2):
        pass

    def seen_state(self, e):
        pass

    def get_action(self, state, env):
        world_state_hash = env.env.get_world_state_hash()
        admissible = env.conn_valid.get(world_state_hash)
        if admissible is None:
            print ("NEED TO GENERATE ACTIONS")
        try:
            admissible = [eval(a.strip()) for a in admissible.decode('cp1252').split('/')]
        except Exception as e:
            print("Exception: {}. Admissible: {}".format(e, admissible))
        
        #print (admissible)
        if admissible:
            return admissible[np.random.choice(len(admissible))].action
        # if self.remaining <= 0:
        #     self.action = env.action_space.sample()
        #     # Note, this is equivalent to selecting an action and then repeating it
        #     # with some probability.
        #     self.remaining = np.random.geometric(1 / self.mean_repeat)
        # self.remaining -= 1
        # return self.action

    def __repr__(self):
        return 'GetActionRandomExplorer()'