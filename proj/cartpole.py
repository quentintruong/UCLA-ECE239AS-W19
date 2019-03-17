from keras.layers import InputLayer, Dense, Activation
from keras.models import clone_model
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.losses import categorical_crossentropy
from keras import backend as K
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from collections import deque
import time
import sys
import os


tf.logging.set_verbosity(tf.logging.ERROR)


def time_run_plot(func, num_iters, bin_width, file_name, verbose):
    if num_iters <= 0:
        return

    start = time.time()
    results = []
    
    for i in range(num_iters):
        sys.stdout.write("\rIteration: %d / %d, Time: %d" % (i, num_iters, time.time() - start))
        sys.stdout.flush()

        results.append(func())
    sys.stdout.write("\r")
    sys.stdout.flush()

    if verbose > 0:
        print("Avg seconds per iteration: {}".format(str((time.time() - start) / num_iters)))
        print("Avg episodes per iteration: {}".format(str(np.mean(results))))
        print()
    
    plt.hist(results, bins=range(min(results), max(results) + bin_width, bin_width))
    plt.xlabel('Episodes')
    plt.ylabel('Games won')
    plt.savefig(file_name)
    plt.close()

def QL(env, buckets, bucket_bounds, epsilon_update, lr_update, discount, max_episodes, verbose):
    def state_to_bucket(buckets, bucket_bounds, states):
        bucket_states = []
        for i in range(len(states)):
            ratio = (bucket_bounds[i] + states[i]) / (2 * bucket_bounds[i])
            new_state = min(buckets[i] - 1, max(0, int(round((buckets[i] - 1) * ratio)))) # Return a state within [0, buckets[i] - 1]
            bucket_states.append(new_state)

        return tuple(bucket_states)


    last_100_games = deque(maxlen=100)

    num_actions = 2
    q_table = np.zeros(buckets + (num_actions,))

    for episode in range(max_episodes):
        epsilon = epsilon_update(episode)
        learning_rate = lr_update(episode)
        done = False
        step_cnt = 0
        state = env.reset()
        state = state_to_bucket(buckets, bucket_bounds, state)

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            next_state = state_to_bucket(buckets, bucket_bounds, next_state)

            # Bellman equation
            q_table[state + (action,)] += learning_rate * (reward + discount * np.max(q_table[next_state]) - q_table[state + (action,)])
            state = next_state
            step_cnt += 1

        if verbose > 0:
            print("Episode: {} steps: {}".format(episode, step_cnt))
        
        last_100_games.append(step_cnt)
        if episode >= 100 and np.mean(last_100_games) >= 195:
            return episode

    return max_episodes

def DQN(env, model, compile, use_target, discount, epsilon_update, target_update_freq, batch_size, replay_memory, max_episodes, verbose):
    replay_buffer = deque(maxlen=replay_memory)
    last_100_games = deque(maxlen=100)

    mainDQN = clone_model(model)
    if use_target:
        targetDQN = clone_model(model)
    else:
        targetDQN = mainDQN
    compile(mainDQN)

    for episode in range(max_episodes):
        epsilon = epsilon_update(episode)
        done = False
        step_cnt = 0
        state = env.reset()

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN.predict(state.reshape(1, 4)))

            next_state, reward, done, _ = env.step(action)

            if done: # if failed, add penalty
                reward = -1

            # Record experiences
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) > batch_size:
                minibatch = np.random.choice(len(replay_buffer), batch_size, replace=False)

                states = np.vstack([replay_buffer[i][0] for i in minibatch])
                actions = np.array([replay_buffer[i][1] for i in minibatch])
                rewards = np.array([replay_buffer[i][2] for i in minibatch])
                next_states = np.vstack([replay_buffer[i][3] for i in minibatch])
                dones = np.array([replay_buffer[i][4] for i in minibatch])

                y = mainDQN.predict(states)  
                y[np.arange(len(states)), actions] = rewards + discount * np.max(targetDQN.predict(next_states), axis=1) * ~dones
                history = mainDQN.fit(x=states, y=y, verbose=0)

            if use_target and step_cnt % target_update_freq == 0:
                targetDQN.set_weights(mainDQN.get_weights()) 

            state = next_state
            step_cnt += 1

        if verbose > 0:
            print("Episode: {} steps: {}".format(episode, step_cnt))

        last_100_games.append(step_cnt)
        if episode >= 100 and np.mean(last_100_games) >= 195:
            return episode

    return max_episodes

def PG(env, model, discount, learning_rate, max_episodes, verbose):
    def discount_and_normalize_rewards(episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * discount + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative
        
        return (discounted_episode_rewards - np.mean(discounted_episode_rewards)) / np.std(discounted_episode_rewards)


    last_100_games = deque(maxlen=100)

    model = clone_model(model)
    action_onehot_placeholder = K.placeholder(shape=(None, 2))
    discount_reward_placeholder = K.placeholder(shape=(None,))
    loss = categorical_crossentropy(action_onehot_placeholder, model.output) * discount_reward_placeholder
    loss = K.mean(loss)
    updates = Adam(lr=learning_rate).get_updates(params=model.trainable_weights, loss=loss)
    fit = K.function(inputs=[model.input, action_onehot_placeholder, discount_reward_placeholder],
                     outputs=[],
                     updates=updates)

    for episode in range(max_episodes):
        done = False
        step_cnt = 0
        episode_states, episode_actions, episode_rewards = [],[],[]
        episode_rewards_sum = 0
        state = env.reset()
        
        while not done:
            action = np.random.choice(2, p=model.predict(state.reshape(1, 4))[0])

            new_state, reward, done, info = env.step(action)

            action_ = np.zeros(2)
            action_[action] = 1.0
            
            episode_states.append(state)
            episode_actions.append(action_)
            episode_rewards.append(reward)
            
            state = new_state
            step_cnt += 1

        fit([episode_states, episode_actions, discount_and_normalize_rewards(episode_rewards)])

        if verbose > 0:
            print("Episode: {} steps: {}".format(episode, step_cnt))

        last_100_games.append(step_cnt)
        if episode >= 100 and np.mean(last_100_games) >= 195:
            return episode

    return max_episodes

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
