from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print("max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25)))")
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
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_ceil1-logd10',
                    verbose=1)

    print("max(0.01, 1 - 0.013 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.013 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-013episode',
                    verbose=1)

    print("max(0.01, 1 - 0.012 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.012 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-012episode',
                    verbose=1)

    print("max(0.01, 1 - 0.011 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.011 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-011episode',
                    verbose=1)

    print("max(0.01, 1 - 0.010 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.010 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-010episode',
                    verbose=1)

    print("max(0.01, 1 - 0.009 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.009 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-009episode',
                    verbose=1)

    print("max(0.01, 1 - 0.008 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.008 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-008episode',
                    verbose=1)

    print("max(0.01, 1 - 0.007 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.007 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-007episode',
                    verbose=1)

    print("max(0.01, 1 - 0.006 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.006 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-006episode',
                    verbose=1)

    print("max(0.01, 1 - 0.005 * episode)")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.005 * episode),
                            discount=1,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=30,
                    bin_width=10,
                    file_name='ql-lr=max01_1-005episode',
                    verbose=1)