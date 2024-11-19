from abc import abstractmethod
import csv
import logging
import numpy as np
import os
from gym import spaces
from .multiagentenv import MultiAgentEnv
import json
import datetime as dt

class SigmoidBase(MultiAgentEnv):
    """
    SigmoidMultiActMultiValAction
    Each agent has the state space only for itself
    """

    def _sig(self, x, scaling, inflection):
        """ Simple sigmoid """
        return 1 / (1 + np.exp(-scaling * (x - inflection)))

    def __init__(self,
                 n_steps: int = 10,
                 n_agents: int = 2,  # agents number
                 n_actions: int = 10,  # action each agent
                 action_vals: tuple = (5, 10),
                 seed: int = 2023,
                 alg_name: str = 'SQAL',
                 open_rand: bool = False,
                 noise: float = 0.0,
                 instance_feats: str = None, # 'instance_feats.csv'
                 slope_multiplier: float = 2,
                 key: str = "Sigmoid",
                 replay_dir=None
                 ) -> None:
        super().__init__()
        self.is_reset = True
        self.replay_dir = replay_dir
        self.replay_his = {'reward': []}
        self.reward_his = []
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.env_seed = seed
        self.alg_name = alg_name
        self.open_rand = open_rand
        print('---', self.n_steps, self.n_agents, self.n_actions, self.env_seed, self.alg_name, self.open_rand)
        action_vals = [self.n_actions for i in range(self.n_agents)]
        self.rng = np.random.RandomState(seed)
        self._c_step = 0
        assert self.n_agents == len(action_vals), (
            f'action_vals should be of length {self.n_agents}.')
        self.shifts = [self.n_steps / 2 for _ in action_vals]
        self.slopes = [-1 for _ in action_vals]
        self.reward_range = (0, 1)
        self._c_step = 0
        self.noise = noise
        self.slope_multiplier = slope_multiplier
        self.action_vals = action_vals
        # budget spent, inst_feat_1, inst_feat_2
        # self._state = [-1 for _ in range(3)]
        # self.action_space = spaces.MultiDiscrete(action_vals)
        self.action_space = [spaces.Discrete(self.n_actions) for _ in range(self.n_agents)]
        '''
        self.action_space = spaces.Box(
            low=0,
            high=n_actions-1,
            shape=(n_agents, )) #todo for dqn
        '''
        if not self.open_rand:
            self.observation_space = [spaces.Box(
                low=np.array([-np.inf for _ in range(1 + self.n_agents * 2)]),
                high=np.array([np.inf for _ in range(1 + self.n_agents * 2)])) for _ in range(self.n_agents)] # todo:0(not 2) for unseen shift and slope
            # self.action_list = self.take_action() #todo for dqn

            self.share_observation_space = [spaces.Box(
                low=np.array([-np.inf for _ in range(1 + self.n_agents * 2)]),
                high=np.array([np.inf for _ in range(1 + self.n_agents * 2)]))]
        else:
            self.observation_space = [spaces.Box(
                low=np.array([-np.inf for _ in range(1 + self.n_agents * 0)]),
                high=np.array([np.inf for _ in range(1 + self.n_agents * 0)])) for _ in
                range(self.n_agents)]  # todo:0(not 2) for unseen shift and slope
            # self.action_list = self.take_action() #todo for dqn

            self.share_observation_space = [spaces.Box(
                low=np.array([-np.inf for _ in range(1 + self.n_agents * 0)]),
                high=np.array([np.inf for _ in range(1 + self.n_agents * 0)]))]

        # initial State and Obs
        self._c_step = 0
        remaining_budget = self.n_steps - self._c_step
        next_state = [remaining_budget]
        self.obs = []

        if not self.open_rand:
            for shift, slope in zip(self.shifts, self.slopes):
                next_state.append(shift)
                next_state.append(slope)
                # self.obs.append([remaining_budget, shift, slope, -1])
                self.obs.append([remaining_budget, shift, slope])
         # todo: unseen shift and slope to add randomness
        # next_state += [-1 for _ in range(self.n_agents)] todo() unseen actions
        self.state = next_state

        self.logger = logging.getLogger(self.__str__())
        self.logger.setLevel(logging.ERROR)
        self._prev_state = None
        self._inst_feat_dict = {}
        self._inst_id = None
        if instance_feats:
            with open(instance_feats, 'r') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    self._inst_feat_dict[int(row['ID'])] = [float(shift) for shift in row['shifts'].split(
                        ",")] + [float(slope) for slope in row['slopes'].split(",")]
                self._inst_id = -1

        # For compatibility With Epymarl
        self.episode_limit = self.n_steps

    def take_action(self):
        action_size = self.n_actions ** self.n_agents
        action_list = []
        for i in range(action_size):
            rest = i
            act = []
            while True:
                act.append(rest % self.n_actions)
                rest = rest // self.n_actions
                if len(act) == self.n_agents:
                    break
            assert(len(act) == self.n_agents)
            action_list.append(act)
        return action_list


    def step(self, action):
        """

        @param action:  List: [x,x,...,x]
        @return: Returns reward, terminated, info
        """
        # action = self.action_list[action] #todo: for dqn
        val = self._c_step
        # action = action.cpu().tolist() # todo: deleted for dqn
        # r = [1 - np.abs(self._sig(val, slope, shift) - (act / (max_act - 1)))
        #     for slope, shift, act, max_act in zip(
        #    self.slopes, self.shifts, action, self.action_vals
        #)]
        r = []
        '''
        r.append(1 - np.abs(self._sig(val, self.slopes[0], self.shifts[0]) - action[0]/(self.action_vals[0]-1)))
        for i in range(1,len(action)):
            if action[i-1] != 0:
                r.append(1 - np.abs(self._sig(val, self.slopes[i], self.shifts[i]) - 2*min(action[i-1]/(self.action_vals[i-1]-1), action[i]/(self.action_vals[i]-1))/(action[i-1]/(self.action_vals[i-1]-1) + action[i]/(self.action_vals[i]-1))))
            else:
                r.append(1 - np.abs(self._sig(val, self.slopes[i], self.shifts[i]) - action[i]/(self.action_vals[i]-1)))
        
        r = np.clip(np.prod(r), 0.0, 1.0)
        
        assert(self.n_agents == 2)
        if action[1] == 0:
            r = 1 - np.abs(self._sig(val, self.slopes[0], self.shifts[0]) - action[0]/(self.action_vals[0]-1))
        elif action[1] == 1:
            r = np.abs(self._sig(val, self.slopes[0], self.shifts[0]) - action[0]/(self.action_vals[0]-1))
        else:
            r = 0
        

        for i in range(len(action)):
            # r.append(1 - np.abs(self._sig(val, self.slopes[i], self.shifts[i]) - sum([action[j] / (self.action_vals[j] - 1) for j in range(i+1)])))
            r.append(1 - np.abs(self._sig(val, self.slopes[i], self.shifts[i]) - sum([action[j]/(self.action_vals[j]-1) for j in range(i,len(action))])))
        '''
        factor = 1
        for i in range(len(action)):
            sig1 = self._sig(val, factor * self.slopes[i], self.shifts[i])
            sig2 = 1 - self._sig(val, factor * self.slopes[i], self.shifts[i])
            if action[i] / (self.action_vals[i] - 1) >= 0.5:
                factor = 10
            else:
                factor = 0.1
            r.append(1 - min(np.abs(sig1 - action[i] / (self.action_vals[i] - 1)), np.abs(sig2 - action[i] / (self.action_vals[i] - 1))))


        r = np.clip(np.prod(r), 0.0, 1.0)

        self.reward_his.append(r)
        remaining_budget = self.n_steps - self._c_step

        next_state = [remaining_budget]

        self.obs = []
        '''
        for shift, slope, a in zip(self.shifts, self.slopes, action):
            next_state.append(shift)
            next_state.append(slope)
            self.obs.append([remaining_budget, shift, slope, a])
        '''
        if not self.open_rand:
            for shift, slope in zip(self.shifts, self.slopes):
                next_state.append(shift)
                next_state.append(slope)
                self.obs.append([remaining_budget, shift, slope])
             # todo unseen shift and slope to add randomness
            # next_state += action todo() unseen actions
        prev_state = self._prev_state
        self.state = next_state

        self.logger.debug("i: (s, a, r, s') / %d: (%s, %s, %5.2f, %2s)", self._c_step - 1, str(prev_state),
                          str(action), r, str(next_state))
        self._c_step += 1
        self._prev_state = next_state

        if self._c_step >= self.n_steps:
            self.save_replay(self.alg_name)
            self.replay_his['reward'].append(self.reward_his)

        return self.state, r, \
            [{'reward': r} for _ in range(self.n_agents)], self._c_step >= self.n_steps, self.get_avail_actions()
        # return self.state, r, self._c_step >= self.n_steps, {'reward': r} # todo:for dqn

    @abstractmethod
    def reset(self):
        pass

    def render(self, mode: str, close: bool = True) -> None:
        pass

    @abstractmethod
    def get_obs(self):
        """ Returns all agent observations in a list """
        pass

    @abstractmethod
    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        pass

    @abstractmethod
    def get_obs_size(self):
        """ Returns the shape of the observation """
        pass

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.state)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1 for i in range(self.n_actions)]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def close(self):
        # self.save_replay('hasac')
        return

    def seed(self, seed):
        pass

    def save_replay(self, alg):
        save_dir = 'results/sigmoid_exp'
        if not os.path.exists(save_dir):
            os.umask(0)
            os.makedirs(save_dir, mode=0o777)
        now_time = dt.datetime.now().strftime("%F %T")
        # token = f"{str(alg)}_n_agents{str(self.n_agents)}_{str(now_time).replace(' ','')}_replay.json"
        token = f"{str(alg)}_n_agents{str(self.n_agents)}_seed{self.env_seed}_IsRand({str(self.open_rand)})_replay.json"
        replay_path = os.path.join(save_dir, token)
        with open(replay_path, 'a') as f:
            # json.dump(self.replay_his, f)
            json.dump(self.reward_his, f)
            f.write('\n')

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return None

