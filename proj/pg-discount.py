from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print("99")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.99,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-discount=99',
                    verbose=1)

    print("95")
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
                    file_name='pg-discount=95',
                    verbose=1)
    
    print("90")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.90,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-discount=90',
                    verbose=1)

    print("85")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.85,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-discount=85',
                    verbose=1)

    print("80")
    model = Sequential()
    model.add(Dense(128, input_shape=(4, ), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    time_run_plot(  func=lambda:
                        PG( env=env, 
                            model=model,
                            discount=0.80,
                            learning_rate=0.007,
                            max_episodes=500,
                            verbose=0),
                    num_iters=40,
                    bin_width=20,
                    file_name='pg-discount=80',
                    verbose=1)
