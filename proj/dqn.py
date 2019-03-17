from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print('adamlr0001')
    model = Sequential()
    model.add(Dense(16, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001)),
                            use_target=True,
                            discount=0.85,
                            epsilon_update=lambda episode : 1. / (episode / 10 + 1),
                            target_update_freq=5,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=1000,
                            verbose=0),
                    num_iters=15,
                    bin_width=20,
                    file_name='dqn-opt=adamlr0001',
                    verbose=1)

    print('adamlr001')
    model = Sequential()
    model.add(Dense(16, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001)),
                            use_target=True,
                            discount=0.85,
                            epsilon_update=lambda episode : 1. / (episode / 10 + 1),
                            target_update_freq=5,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=1000,
                            verbose=0),
                    num_iters=15,
                    bin_width=20,
                    file_name='dqn-opt=adamlr001',
                    verbose=1)

    print('adamlr005')
    model = Sequential()
    model.add(Dense(16, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.005)),
                            use_target=True,
                            discount=0.85,
                            epsilon_update=lambda episode : 1. / (episode / 10 + 1),
                            target_update_freq=5,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=1000,
                            verbose=0),
                    num_iters=15,
                    bin_width=20,
                    file_name='dqn-opt=adamlr005',
                    verbose=1)

    print('adamlr001decay999999')
    model = Sequential()
    model.add(Dense(16, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, decay=0.999999)),
                            use_target=True,
                            discount=0.85,
                            epsilon_update=lambda episode : 1. / (episode / 10 + 1),
                            target_update_freq=5,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=1000,
                            verbose=0),
                    num_iters=15,
                    bin_width=20,
                    file_name='dqn-opt=adamlr001decay999999',
                    verbose=1)

    print('adamlr001amsgrad')
    model = Sequential()
    model.add(Dense(16, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, amsgrad=True)),
                            use_target=True,
                            discount=0.85,
                            epsilon_update=lambda episode : 1. / (episode / 10 + 1),
                            target_update_freq=5,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=1000,
                            verbose=0),
                    num_iters=15,
                    bin_width=20,
                    file_name='dqn-opt=adamlr001amsgrad',
                    verbose=1)

    print('nadam001')
    model = Sequential()
    model.add(Dense(16, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Nadam(lr=0.001)),
                            use_target=True,
                            discount=0.85,
                            epsilon_update=lambda episode : 1. / (episode / 10 + 1),
                            target_update_freq=5,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=1000,
                            verbose=0),
                    num_iters=15,
                    bin_width=20,
                    file_name='dqn-opt=nadam001',
                    verbose=1) 

    print('adam0001')
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001)),
                            use_target=True,
                            discount=0.99,
                            epsilon_update=lambda episode : max(0.01, 1 - episode * 0.08),
                            target_update_freq=100,
                            batch_size=128,
                            replay_memory=50000,
                            max_episodes=500,
                            verbose=1),
                    num_iters=15,
                    bin_width=20,
                    file_name='dqn-opt=adam0001',
                    verbose=1) 

    print('adam0101')
    model = Sequential()
    model.add(Dense(64, input_shape=(4, ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))
    time_run_plot(  func=lambda:
                        DQN(env=env, 
                            model=model,
                            compile=lambda obj : obj.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001)),
                            use_target=True,
                            discount=0.99,
                            epsilon_update=lambda episode : max(0.01, 1 - episode * 0.01),
                            target_update_freq=100,
                            batch_size=64,
                            replay_memory=50000,
                            max_episodes=500,
                            verbose=1),
                    num_iters=10,
                    bin_width=20,
                    file_name='dqn-opt=adam0001tanh',
                    verbose=1) 
