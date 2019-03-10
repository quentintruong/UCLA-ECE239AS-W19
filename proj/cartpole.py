import gym
import numpy as np
from matplotlib import pyplot

def step_random():
    env = gym.make('CartPole-v1')
    env.reset()
    done = False
    while not done:
        env.render()
        new_s, r, done, _ = env.step(env.action_space.sample()) # take a random action
    
    env.close()

def dqn():
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 5)))
    model.add(Dense(10, input_shape=(5,), activation='sigmoid'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    num_episodes = 1

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
            target_vec = model.predict(np.identity(5)[s:s + 1])[0]
            target_vec[a] = target
            print(np.identity(5)[s:s+1])
            print(target_vec.reshape(-1,2))
            model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)

if __name__ == '__main__':
    #step_random()
    dqn()
