from .base_env import SigmoidBase


class SigmoidMultiActMultiValActionState(SigmoidBase):
    """
    Each agent shares the same state
    """

    def reset(self):
        """ Returns initial observations and states"""
        self.reward_his = []
        # if self.is_reset:
        if self._inst_feat_dict:
            self._inst_id = (self._inst_id + 1) % len(self._inst_feat_dict)
            self.shifts = self._inst_feat_dict[self._inst_id][:self.n_agents]
            self.slopes = self._inst_feat_dict[self._inst_id][self.n_agents:]
        else:
            if not self.open_rand:
                self.shifts = self.rng.normal(
                    self.n_steps / 2, self.n_steps / 4, self.n_agents)
                self.slopes = self.rng.choice([-1, 1], self.n_agents) * self.rng.uniform(
                    size=self.n_agents) * self.slope_multiplier
            else:
                self.shifts = self.rng.normal(
                    0, self.n_steps / 4, self.n_agents)
                self.slopes = self.rng.choice([-1, 1], self.n_agents)
                # todo:add randomness
            # self.is_reset = False
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
        self._prev_state = None
        self.logger.debug(
            "i: (s, a, r, s') / %d: (%2d, %d, %5.2f, %2d)", -1, -1, -1, -1, -1)
        # return self.get_obs(), self.state
        return self.state, self.get_obs()

    def get_obs(self):
        """ Returns all agent observations in a list """
        # return self.state #todo: for dqn
        return [self.state for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.state

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.state)
