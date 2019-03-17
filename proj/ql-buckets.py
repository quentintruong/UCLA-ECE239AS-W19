from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print("(1, 1, 5, 2)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 5, 2), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=1152',
                    verbose=1)

    print("(1, 1, 6, 3)")
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
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=1163',
                    verbose=1)

    print("(1, 2, 6, 3)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 2, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=1263',
                    verbose=1)

    print("(1, 3, 6, 3)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 3, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=1363',
                    verbose=1)

    print("(2, 1, 6, 3)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(2, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=2163',
                    verbose=1)

    print("(3, 1, 6, 3)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(3, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=3163',
                    verbose=1)

    print("(2, 2, 6, 3)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(2, 2, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=2263',
                    verbose=1)
    
    print("(3, 3, 6, 3)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(3, 3, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : max(0.1, min(1, 1.0 - np.log10((episode + 1) / 25))),
                            lr_update=lambda episode : max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25))),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-buckets=3363',
                    verbose=1)
