import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.stats
# Function to plot result


max_episodes = 1000
x_range = 1000

# single plots: Exp Avg reward
plt.figure(figsize=(16, 8))
file_type1 = "length_of_episodes"
data_mean_jumpstart_baseline = []
data_mean_final_reward_baseline = []

def plot_baseline():
    directory = 'optimality_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_None_optimality_optimal'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="Baseline")


def plot_extra_action():
    directory = 'optimality_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1


    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_optimal'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with optimal demonstration")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_suboptimal_left'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with suboptimal_left demonstration")
    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_suboptimal_right'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with suboptimal_right demonstration")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with noisy demonstration")


def plot_value_bonus():
    directory = 'optimality_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_optimal'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with optimal demonstration")

    ###############################

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_suboptimal_left'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with suboptimal_left demonstration")

    ###############################

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_suboptimal_right'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with suboptimal_right demonstration")


    ###############################

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0, label="value bonus with noisy demonstration")


def plot_PPR():
    directory = 'optimality_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_optimal'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="probabilistic policy reuse with optimal demonstration")

    ###############################

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_suboptimal_left'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="probabilistic policy reuse with suboptimal_left demonstration")

    ###############################

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_suboptimal_right'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="probabilistic policy reuse with suboptimal_right demonstration")

    ###############################

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    data_mean, data_std_err, plt_x_legend = calculate_mean(data)
    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="probabilistic policy reuse with noisy demonstration")


def plot_extra_action_noisy():
    directory = 'noisy_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1


    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.1'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with noisy prob 0.1")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.3'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with noisy prob 0.3")
    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.5'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with noisy prob 0.5")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.7'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with noisy prob 0.7")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.9'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="extra action with noisy prob 0.9")


def plot_value_bonus_noisy():
    directory = 'noisy_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1


    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.1'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with noisy prob 0.1")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.3'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with noisy prob 0.3")
    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.5'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with noisy prob 0.5")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.7'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with noisy prob 0.7")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.9'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="value bonus with noisy prob 0.9")


def plot_PPR_noisy():
    directory = 'noisy_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1


    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.1'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="PPR with noisy prob 0.1")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.3'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="PPR with noisy prob 0.3")
    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.5'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="PPR with noisy prob 0.5")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.7'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="PPR with noisy prob 0.7")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.9'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    data_mean, data_std_err, plt_x_legend = calculate_mean(data)

    plt.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    plt.plot(plt_x_legend, data_mean, linewidth=1.0,label="PPR with noisy prob 0.9")

def calculate_mean(data):

    check_point = 20
    new_data = np.zeros([50, 1000//check_point])
    for i in range(0, len(data)):
        temp = data[i]
        new_data[i] = np.mean(temp.reshape(-1, check_point), axis=1)
    data_mean = np.mean(new_data, axis=0)
    data_std_err = np.std(new_data, axis=0) / np.sqrt(len(new_data))
    data_std_err = data_std_err
    plt_x_legend = np.array(range(0, 1000))
    plt_x_legend = np.mean(plt_x_legend.reshape(-1, check_point), axis=1)

    return data_mean, data_std_err, plt_x_legend


def calculate_baseline_t_statistic():
    directory = 'optimality_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_None_optimality_optimal'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    for i in range(0,len(data)):
        data_mean_jumpstart_baseline.append(np.average(data[i][100:300]))
        data_mean_final_reward_baseline.append(np.average(data[i][800:1000]))


def calculate_extra_action_t_statistic():
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1
    directory = 'optimality_result'
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_optimal'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Optimal Extra Action: ")
    print("JumpStart: ",  jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_suboptimal_left'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Suboptimal-Left Extra Action: ")
    print("JumpStart: ",  jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_suboptimal_right'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Suboptimal-Right Extra Action: ")
    print("JumpStart: ",  jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy Extra Action: ")
    print("JumpStart: ",  jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")


def calculate_value_bonus_t_statistic():
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2 ** -7
    critic_ss = 2 ** -4
    gamma = 1
    directory = 'optimality_result'

    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_optimal'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Optimal value_bonus: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_suboptimal_left'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Suboptimal-Left value_bonus: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_suboptimal_right'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Suboptimal-Right value_bonus: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy value_bonus: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")


def calculate_PPR_t_statistic():
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2 ** -7
    critic_ss = 2 ** -4
    gamma = 1
    directory = 'optimality_result'
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_optimal'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Optimal probabilistic_policy_reuse: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_suboptimal_left'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Suboptimal-Left probabilistic_policy_reuse: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_suboptimal_right'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Suboptimal-Right probabilistic_policy_reuse: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data
    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy probabilistic_policy_reuse: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")


def calculate_extra_action_noisy_t_statistic():
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1
    directory = 'noisy_result'

    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.1'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy extra action with N = 0.1: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.3'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy extra action with N = 0.3: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")
    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.5'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy extra action with N = 0.5: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.7'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy extra action with N = 0.7: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_extra_action_optimality_noisy_noisy_prob_0.9'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy extra action with N = 0.9: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")


def calculate_value_bonus_noisy_t_statistic():
    directory = 'noisy_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1


    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.1'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy value bonus with N = 0.1: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.3'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy value bonus with N = 0.3: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")
    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.5'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy value bonus with N = 0.5: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.7'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy value bonus with N = 0.7: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_value_bonus_optimality_noisy_noisy_prob_0.9'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy value bonus with N = 0.9: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")


def calculate_PPR_noisy_t_statistic():
    directory = 'noisy_result'
    num_tilings = 32
    num_tiles = 64
    actor_ss = 2**-7
    critic_ss=2**-4
    gamma = 1


    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.1'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy PPR with N = 0.1: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.3'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy PPR with N = 0.3: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")
    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.5'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy PPR with N = 0.5: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.7'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy PPR with N = 0.7: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

    ###############################
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_gamma_{}_max_episodes_{}_human_demonstration_option_probabilistic_policy_reuse_optimality_noisy_noisy_prob_0.9'.format(num_tilings, num_tiles, actor_ss, critic_ss, gamma, max_episodes)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    data = -data

    jumpstart, final_reward, jumpstart_p, final_reward_p = calculate_stat(data)
    print("For Noisy PPR with N = 0.9: ")
    print("JumpStart: ", jumpstart, " P_value: ", jumpstart_p)
    print("Final reward: ", final_reward, " P_value: ", final_reward_p)
    print("\n")

def calculate_stat(data):
    data_mean_jumpstart = []
    data_mean_final_reward = []
    for i in range(0,len(data)):
        data_mean_jumpstart.append(np.average(data[i][100:300]))
        data_mean_final_reward.append(np.average(data[i][800:1000]))
    jumpstart_stat, jumpstart_p= scipy.stats.ttest_ind(data_mean_jumpstart, data_mean_jumpstart_baseline)
    final_reward_stat, final_reward_p = scipy.stats.ttest_ind(data_mean_final_reward, data_mean_final_reward_baseline)
    temp1 = np.average(data_mean_jumpstart) - np.average(data_mean_jumpstart_baseline)
    temp2 = np.average(data_mean_final_reward) - np.average(data_mean_final_reward_baseline)
    return round(temp1, 2), round(temp2, 2), jumpstart_p, final_reward_p

def main_plot():
    plot_baseline()
    #plot_extra_action()
    #plot_value_bonus()
    #plot_PPR()
    #plot_extra_action_noisy()
    #plot_value_bonus_noisy()
    plot_PPR_noisy()
    plt.ylim(-10000, 0)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel('Number of Episodes', fontsize=20)
    plt.ylabel('Return of the Episode', fontsize=20)
    plt.xlim([0, 1100])
    #plt.legend(loc="best", fontsize=10)
    # plt.title("Return of episodes Softmax Actor Critic (50 Runs)", fontsize=16,
    #               fontweight='bold',
    #               y=1.03)

    plt.tight_layout()
    # plt.savefig('AC_only.png')
    plt.show()


def main_statistic():
    calculate_baseline_t_statistic()

    calculate_extra_action_t_statistic()
    calculate_extra_action_noisy_t_statistic()

    calculate_value_bonus_t_statistic()
    calculate_value_bonus_noisy_t_statistic()

    calculate_PPR_t_statistic()
    calculate_PPR_noisy_t_statistic()






#main_plot()
main_statistic()
