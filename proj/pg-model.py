from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print("32")
    model = Sequential()
    model.add(Dense(32, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=10,
                    bin_width=20,
                    file_name='pg-model=32',
                    verbose=1)

    print("64")
    model = Sequential()
    model.add(Dense(64, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=10,
                    bin_width=20,
                    file_name='pg-model=64',
                    verbose=1)

    print("128")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=10,
                    bin_width=20,
                    file_name='pg-model=128',
                    verbose=1)

    print("32,32")
    model = Sequential()
    model.add(Dense(32, input_shape=(4, ), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=10,
                    bin_width=20,
                    file_name='pg-model=32,32',
                    verbose=1)

    print("64,64")
    model = Sequential()
    model.add(Dense(64, input_shape=(4, ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=10,
                    bin_width=20,
                    file_name='pg-model=64,64',
                    verbose=1)

    print("128,128")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=10,
                    bin_width=20,
                    file_name='pg-model=128,128',
                    verbose=1)
