import gym
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import load_model
from keras.layers import InputLayer, Dense, Activation

def naive_sum_reward_agent(env, num_episodes=500):
    # this is the table that will hold our summated rewards for
    # each action in each state
    r_table = np.zeros((5, 2))
    for g in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.sum(r_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with highest cummulative reward
                a = np.argmax(r_table[s, :])
            new_s, r, done, _ = env.step(a)
            r_table[s, a] += r
            s = new_s
    return r_table

def q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95
    lr = 0.8
    for i in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.sum(q_table[s,:]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with largest q value in state s
                a = np.argmax(q_table[s, :])
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr*(y*np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table

def eps_greedy_q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s, :])
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table

def train_dqn(env):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 5)))
    model.add(Dense(10, input_shape=(5,), activation='sigmoid'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    num_episodes = 1000

    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            new_s, r, done, _ = env.step(a)
            target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))

            #print(model.predict(np.identity(5)[s:s + 1]))
            target_vec = model.predict(np.identity(5)[s:s + 1])[0]
            target_vec[a] = target
            #print(np.identity(5)[s:s+1])
            #print(target_vec.reshape(-1,2))
            model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    pyplot.plot(r_avg_list)
    pyplot.show()

    model.save('mymodel.h5')

def run_dqn(env):
    model = load_model('mymodel.h5')
    
    r_avg_list = []
    for i in range(100):
        s = env.reset()
        done = False
        r_sum = 0
        while not done:
            a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            s, r, done, _ = env.step(a)
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    pyplot.plot(r_avg_list)
    pyplot.show()

def step_random(env):
    r_avg_list = []

    for i in range(1000):
        s = env.reset()
        done = False
        r_sum = 0
        while not done:
            a = np.random.randint(0, 2)
            s, r, done, _ = env.step(a)
            r_sum += r
        r_avg_list.append(r_sum / 1000)
    pyplot.plot(r_avg_list)
    pyplot.show()

if __name__ == '__main__':
    #step_random(gym.make('NChain-v0'))
    train_dqn(gym.make('NChain-v0'))
    #run_dqn(gym.make('NChain-v0'))
