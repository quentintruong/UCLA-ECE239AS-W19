from keras.layers import InputLayer, Dense, Activation
from keras.models import clone_model
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras import backend as K
import gym
import matplotlib.pyplot as plt
import numpy as np

from collections import deque
import time


def time_run_plot(func, num_iters, bin_width, file_name, verbose):
    if num_iters <= 0:
        return

    start = time.time()
    results = []
    
    for i in range(num_iters):
        results.append(func())

    if verbose > 0:
        print("Avg seconds per iteration: {}".format(str((time.time() - start) / num_iters)))
        print("Avg episodes per iteration: {}".format(str(np.mean(results))))
    
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

            #if done: # if failed, add penalty
            #    reward = -1

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

    ### QL ###
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([env.observation_space.high[0], .5, env.observation_space.high[2], np.radians(50)]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.1, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=0,
                    bin_width=10,
                    file_name='a',
                    verbose=1)

    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(3, 3, 6, 3), 
                            bucket_bounds=np.array([env.observation_space.high[0], .5, env.observation_space.high[2], np.radians(50)]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.1, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=0,
                    bin_width=10,
                    file_name='b',
                    verbose=1)

    ### DQN ###
    model = Sequential()
    model.add(Dense(16, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001)),
                            use_target=True,
                            discount=0.90,
                            epsilon_update=lambda episode : 1. / (episode / 10 + 1),
                            target_update_freq=5,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=1000,
                            verbose=1),
                    num_iters=1,
                    bin_width=20,
                    file_name='b',
                    verbose=1)

    ### PG ###
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    #model.add(Dense(2, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.9,
                            learning_rate=0.009,
                            max_episodes=1000,
                            verbose=1),
                    num_iters=0,
                    bin_width=20,
                    file_name='a',
                    verbose=1)
