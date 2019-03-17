from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    
    print("max(0.01, min(1, 1.0 - np.log10((episode + 1) / 10)))")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.01, min(1, 1.0 - np.log10((episode + 1) / 10))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=max01_ceil1-logd10',
                    verbose=1)
    
    print("max(0.01, min(1, 1.0 - np.log10((episode + 1) / 15)))")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.01, min(1, 1.0 - np.log10((episode + 1) / 15))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=max01_ceil1-logd15',
                    verbose=1)
    
    print("max(0.01, min(1, 1.0 - np.log10((episode + 1) / 20)))")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.01, min(1, 1.0 - np.log10((episode + 1) / 20))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=max01_ceil1-logd20',
                    verbose=1)

    print("max(0.01, min(1, 1.0 - np.log10((episode + 1) / 25)))")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.01, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=max01_ceil1-logd25',
                    verbose=1)

    print("max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25)))")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=max1_ceil1-logd25',
                    verbose=1)

    print("0.1")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 0.1,
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=1',
                    verbose=1)

    print("1 / (episode + 1)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 / (episode + 1),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epsilon=inverse',
                    verbose=1)

    print("max(0.01, 0.983 ** episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.01, 0.983 ** episode),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=max01_983esp',
                    verbose=1)

    print("1 - 0.007 * episode")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=1-007episode',
                    verbose=1)

    print("max(0.01, 1 - 0.007 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.01, 1 - 0.007 * episode),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-epislon=max01_1-007episode',
                    verbose=1)
