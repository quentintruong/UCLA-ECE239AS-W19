from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')


    print("011")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.011,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-lr=011',
                    verbose=1)

    print("009")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.009,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-lr=009',
                    verbose=1)

    print("007")
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
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-lr=007',
                    verbose=1)

    print("005")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.005,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-lr=005',
                    verbose=1)

    print("003")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.95,
                            learning_rate=0.003,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-lr=003',
                    verbose=1)
