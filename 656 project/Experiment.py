#import plot_AC
import os
from tqdm import tqdm
from numpy import unravel_index
from AC_agent import *
from ExtendedMountainCarEnv import*
import matplotlib.pyplot as plt
import json
import copy

def run_experiment(environment, agent, agent_parameters, experiment_parameters):
    env_info = {}
    # results to save
    length_of_episode_array = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_episodes"]))
    # using tqdm we visualize progress bars
    num_tilings = agent_parameters["num_tilings"]
    num_tiles = agent_parameters["num_tiles"]
    actor_ss = agent_parameters["actor_step_size"]
    critic_ss = agent_parameters["critic_step_size"]
    gamma = agent_parameters["gamma"]
    human_demonstration_option = agent_parameters.get("human_demonstration_option")
    optimality = agent_parameters.get("optimality")
    noisy_prob = agent_parameters.get("noisy_prob")

    for run in tqdm(range(0, experiment_parameters["num_runs"])):
        env_info["seed"] = run
        agent_parameters["seed"] = run
        current_agent = agent()
        current_agent.agent_init(agent_info=agent_parameters)
        environment.seed(run)
        for episode in range(0,experiment_parameters["max_episodes"]):
            length_of_episode = 0
            observation = environment.reset()
            action = current_agent.agent_start(observation)
            while True:
                #environment.render()
                observation, reward, done, infos = environment.step(action)
                length_of_episode += 1
                if done: #at the begining, we do intialization
                     current_agent.agent_end(reward)
                     break
                else:
                    action = current_agent.agent_step(reward, observation)
            print(length_of_episode)
            print("Currently the %d th episode"% episode)
            length_of_episode_array[run][episode] = length_of_episode
        plt.plot(range(0,experiment_parameters["max_episodes"]), length_of_episode_array[run])
        #plt.show()
        if not os.path.exists('optimality_result'):
            os.makedirs('optimality_result')

        save_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_{}_optimality_{}_noisy_prob_{}'.format(
                            num_tilings, num_tiles, actor_ss, critic_ss, gamma, experiment_parameters["max_episodes"], human_demonstration_option, optimality, noisy_prob)
        length_of_episode_array_filename = "optimality_result/{}_length_of_episodes.npy".format(save_name)
        np.save(length_of_episode_array_filename, length_of_episode_array)


def load_obj(name):
    with open('human_demonstration_data/' + name + '.json', 'rb') as f:
        return json.load(f)

def AC_with_Human():
    # This function runs the experiment for some specific parameters
    # Experiment parameters
    max_episodes = 1000
    # This function runs the experiment for some specific parameters
    # Experiment parameters
    experiment_parameters = {
        "max_episodes" : max_episodes,
        "num_runs" : 50
    }
    human_demonstration_data = load_obj("action_list")

    # Environment parameters
    # Agent parameters
    #   Each element is an array because we will be later sweeping over multiple values
    # actor and critic step-sizes are divided by num. tilings inside the agent
    agent_parameters = {
        "num_tilings": 32,
        "num_tiles": 64,
        "actor_step_size": 2**-7,
        "critic_step_size": 2**-4,
        "gamma": 1,
        "num_actions": 3,
        "human_demonstration_data": human_demonstration_data,
        "human_demonstration_option": "probabilistic_policy_reuse",
        "optimality": "noisy",
        "B": 5,
        "C": 100,
        "Sai": 0.99,
        "Sai_discount": 0.95,
        "noisy_prob" : 0.3,
        "auto_solver_threshold": 1e-6,
        "iht_size": 4096*80
    }

    current_env = ExtendedMountainCarEnv()
    current_agent = ActorCriticSoftmaxAgent

    run_experiment(current_env, current_agent, agent_parameters, experiment_parameters)


#AC_only()
AC_with_Human()

