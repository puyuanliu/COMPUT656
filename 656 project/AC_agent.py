# -----------------------------------------------------------
# Implementation of a one-step softmax actor-critic (episodic)
# agent with Extended Mountain Car environment.
#
# (C) 2020 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# email puyuan@ualberta.ca
# -----------------------------------------------------------

import numpy as np
from MC_support import *
import math

class ActorCriticSoftmaxAgent():
    def __init__(self):
        """
        Initialization of the ActorCritic Agent. All values are set to None so they can
        be initialized in the agent_init method.
        """
        self.rand_generator = None
        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None
        self.tc = None
        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None
        self.actions = None
        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
        self.gamma = None
        self.human_demonstration_option = None
        self.human_demonstration_data = None
        self.B = None
        self.C = None
        self.Sai = None
        self.Sai_discount = None
        self.Sai_h = None
        self.episode_counter = None
        self.I = None
        self.optimality = None
        self.noisy_prob = None
        self.auto_solver_threshold = None

    def agent_init(self, agent_info={}):
        """ Initialize the agent with given parameters.
        Args:
            agent_info (dict): a dictionary of the agent parameters
        """

        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        iht_size = agent_info.get("iht_size") # get required parameters from the dictionary
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        self.tc = ExtendedMountainCarTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles) # initialize the tile coder

        self.actor_step_size = agent_info.get("actor_step_size") / num_tilings
        self.critic_step_size = agent_info.get("critic_step_size") / num_tilings
        self.gamma = agent_info.get("gamma", 0.99)
        self.human_demonstration_option = agent_info.get("human_demonstration_option")
        self.human_demonstration_data = agent_info.get("human_demonstration_data")
        self.B = agent_info.get("B", 10)
        self.C = agent_info.get("C", 100)
        self.episode_counter = 0
        self.Sai = agent_info.get("Sai", 0.99)
        self.Sai_discount = agent_info.get("Sai_discount", 0.95)
        self.optimality = agent_info.get("optimality")
        self.noisy_prob = agent_info.get("noisy_prob")
        self.auto_solver_threshold = agent_info.get("auto_solver_threshold")
        self.actions = list(range(agent_info.get("num_actions")))

        if self.human_demonstration_option == "extra_action":
            self.actions = list(range(agent_info.get("num_actions")+1))
            self.actor_w = np.zeros((len(self.actions), iht_size))
            self.critic_w = np.zeros(iht_size)
        else:
            self.actions = list(range(agent_info.get("num_actions")))
            self.actor_w = np.zeros((len(self.actions), iht_size))
            self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_policy(self, active_tiles, state):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder
        Returns:
            The action selected according to the policy
        """
        softmax_prob = self.compute_softmax_prob(self.actor_w, active_tiles)
        self.softmax_prob = softmax_prob
        # compute softmax probability
        position, velocity = state

        # Get the correct human demonstration action according to the demonstration option
        if self.optimality == "optimal":
            human_action = self.get_optimal_action(position, velocity)
        elif self.optimality == "suboptimal_left":
            human_action = self.get_suboptimal_left_action(position, velocity)
        elif self.optimality == "suboptimal_right":
            human_action = self.get_suboptimal_right_action(position, velocity)
        elif self.optimality == "noisy":
            human_action = self.get_noisy_action(position, velocity)



        if self.human_demonstration_option == "probabilistic_policy_reuse":
            if self.rand_generator.random() < 1 - self.Sai:
                chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
            else:
                chosen_action = human_action

        elif self.human_demonstration_option == "value_bonus":
            if self.episode_counter < self.C:
                chosen_action = human_action
            else:
                softmax_prob = self.compute_value_bonus_softmax_prob(self.actor_w, active_tiles, human_action)
                self.softmax_prob = softmax_prob
                chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)

        elif self.human_demonstration_option == "extra_action":
            if self.episode_counter < self.C:
                chosen_action = human_action
            else:
                chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
            # Handle the extra action
            if chosen_action == 3:
                # If the chosen action is the pseudo action
                chosen_action = human_action
        elif self.human_demonstration_option == "graphical_illustration":
            chosen_action = human_action
        else:
            chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
        # save softmax_prob as it will be useful later when updating the Actor

        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        position, velocity = state
        self.Sai_h = self.Sai
        ### Use self.tc to get active_tiles using angle and ang_vel (2 lines)
        # set current_action by calling self.agent_policy with active_tiles

        active_tiles = self.tc.get_tiles(position, velocity)
        current_action = self.agent_policy(active_tiles, state)

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)
        self.I = 1
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        position, velocity = state
        self.Sai_h *= self.Sai_discount
        active_tiles = self.tc.get_tiles(position, velocity)
        pre_state_value = sum(self.critic_w[self.prev_tiles])
        current_state_value = sum(self.critic_w[active_tiles])
        if pre_state_value > 1e5 or pre_state_value < -1e5:
            print(pre_state_value)
        delta = reward + self.gamma*current_state_value - pre_state_value
        self.critic_w[self.prev_tiles] += self.critic_step_size * self.I* delta

        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * self.I * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * self.I * delta * (0 - self.softmax_prob[a])

        current_action = self.agent_policy(self.prev_tiles, state)

        self.prev_tiles = active_tiles
        self.last_action = current_action
        self.I = self.gamma*self.I
        return self.last_action

    def agent_end(self, reward):
        """Last step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
        Returns:
            None
        """
        pre_state_value = sum(self.critic_w[self.prev_tiles])
        delta = reward - pre_state_value
        self.Sai *= self.Sai_discount # we discount sai overtime
        self.critic_w[self.prev_tiles] += self.critic_step_size * self.I * delta
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * self.I * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * self.I * delta * (0 - self.softmax_prob[a])
        self.episode_counter+=1
        self.I = self.gamma * self.I

    def compute_softmax_prob(self, actor_w, tiles):
        """
        Computes softmax probability for all actions

        Args:
        actor_w - np.array, an array of actor weights
        tiles - np.array, an array of active tiles

        Returns:
        softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
        """

        state_action_preferences = []
        for i in range(0, len(self.actions)):
            state_action_preferences.append(actor_w[i][tiles].sum())
        # if max(np.abs(state_action_preferences)) > 10:
        #     print(max(np.abs(state_action_preferences)))
        c = np.max(state_action_preferences)
        numerator = []
        for i in range(0, len(state_action_preferences)):
            numerator.append(np.exp(state_action_preferences[i] - c))

        denominator = np.sum(numerator)

        softmax_prob = numerator / denominator

        return softmax_prob

    def compute_value_bonus_softmax_prob(self, actor_w, tiles, human_action):
        """
        Computes softmax probability for all actions after adding the bonus value to certain action

        Args:
        actor_w - np.array, an array of actor weights
        tiles - np.array, an array of active tiles
        human_action - int, action selected by human and going to be added with bonus value
        Returns:
        softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
        """

        state_action_preferences = []
        for i in range(0, 3):
            state_action_preferences.append(actor_w[i][tiles].sum())
        state_action_preferences[human_action] += self.B  # add value to the human-selected action.
        c = np.max(state_action_preferences)
        numerator = []
        for i in range(0, len(state_action_preferences)):
            numerator.append(np.exp(state_action_preferences[i] - c))

        denominator = np.sum(numerator)

        softmax_prob = numerator / denominator
        return softmax_prob

    def get_optimal_action(self, position, velocity):
        # This function returns the explict action to do to reach the goal (serves as optimal human action)
        goal_position = 0.1
        position_criteria = 0.01
        velocity_criteria = 0.001
        gravity = 0.00015
        test_velocity = velocity
        counter = 0
        target_energy = (np.sin(3 * goal_position)/3)*gravity
        current_energy = (math.sin(3 * position)/3)*gravity + 0.5*velocity**2
        if abs(current_energy - target_energy) < self.auto_solver_threshold:
            return 1
        elif current_energy - target_energy > 0:
            # need less energy, accelerate right (2) if velocity negative, accelerate left (0) if positive
            if velocity < 0:
                return 2
            else:
                return 0
        else:
            # need more energy, accelerate left (0) if velocity negative, accelerate right (2) if positive
            if velocity < 0:
                return 0
            else:
                return 2

    def get_suboptimal_left_action(self, position, velocity):
        # This function returns the explict action to do to reach the goal (serves as optimal human action)
        goal_position = 0.1
        position_criteria = 0.01
        velocity_criteria = 0.001
        gravity = 0.00015
        test_velocity = velocity
        counter = 0
        target_energy = (np.sin(3 * goal_position)/3)*gravity
        current_energy = (math.sin(3 * position)/3)*gravity + 0.5*velocity**2
        if abs(current_energy - target_energy) < self.auto_solver_threshold:
            return 1
        elif current_energy - target_energy > 0:
            # need less energy, accelerate right (2) if velocity negative, accelerate left (0) if positive
            if velocity < 0:
                return 1 # return do nothing option when it's right
            else:
                return 0 # return the correct option when it's left
        else:
            # need more energy, accelerate left (0) if velocity negative, accelerate right (2) if positive
            if velocity < 0:
                return 0 # return the correct option when it's left
            else:
                return 1 # return do nothing option when it's right

    def get_suboptimal_right_action(self, position, velocity):
        # This function returns the explict action to do to reach the goal (serves as optimal human action)
        goal_position = 0.1
        position_criteria = 0.01
        velocity_criteria = 0.001
        gravity = 0.00015
        test_velocity = velocity
        counter = 0
        target_energy = (np.sin(3 * goal_position)/3)*gravity
        current_energy = (math.sin(3 * position)/3)*gravity + 0.5*velocity**2
        if abs(current_energy - target_energy) < self.auto_solver_threshold:
            return 1
        elif current_energy - target_energy > 0:
            # need less energy, accelerate right (2) if velocity negative, accelerate left (0) if positive
            if velocity < 0:
                return 2 # return the correct option when it's right
            else:
                return 1
        else:
            # need more energy, accelerate left (0) if velocity negative, accelerate right (2) if positive
            if velocity < 0:
                return 1
            else:
                return 2 # return the correct option when it's right

    def get_noisy_action(self, position, velocity):
        if self.rand_generator.random() < self.noisy_prob:
            # we have noisy_prob probability to select noisy data
            return self.rand_generator.choice(self.actions[0:3])
        else:
            return self.get_optimal_action(position, velocity)

