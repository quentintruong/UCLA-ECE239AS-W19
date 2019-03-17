from cartpole import *


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print("1")
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
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-discount=1',
                    verbose=1)

    print("999")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.013 * episode),
                            discount=0.999,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-discount=999',
                    verbose=1)

    print("99")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.013 * episode),
                            discount=0.99,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-discount=99',
                    verbose=1)

    print("95")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.013 * episode),
                            discount=0.95,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-discount=95',
                    verbose=1)

    print("90")
    time_run_plot(  func=lambda:
                        QL( env=env,
                            buckets=(1, 1, 6, 3), 
                            bucket_bounds=np.array([1, 1, 0.5, 1]),
                            epsilon_update=lambda episode : 1 - 0.007 * episode,
                            lr_update=lambda episode : max(0.01, 1 - 0.013 * episode),
                            discount=0.90,
                            max_episodes=1000,
                            verbose=0,
                        ),
                    num_iters=60,
                    bin_width=10,
                    file_name='ql-discount=90',
                    verbose=1)