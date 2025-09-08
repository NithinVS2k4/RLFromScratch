import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import time

import torch
import gymnasium as gym

def animate_policy(env, policy, FPS: int = 12, do_truncate: bool = True):
    '''
    Animates a learned policy using the environment's render() method.
    
    Args:
        env: OpenAI Gym environment
        policy: A callable that takes in a state and returns a probability distribution over actions
        FPS (int): Frames per second of the animation
        do_truncate (bool): If True, truncate the animation at episode end or after 100 steps
    '''

    figure_size = (5, 5)
    
    env_id = env.unwrapped.spec.id

    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        if isinstance(env.observation_space, gym.spaces.Discrete):
            s_dims = env.observation_space.n
            if isinstance(policy, np.ndarray):
                policy_arr = policy
                policy = lambda s: torch.tensor(policy_arr[s])
        else:
            s_dims = env.observation_space.shape[0] 
        a_dims = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        is_discrete = False
        if isinstance(env.observation_space, gym.spaces.Discrete):
            s_dims = env.observation_space.n
        else:
            s_dims = env.observation_space.shape[0] 
        a_dims = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Unsupported action space {type(env.action_space)}")

    s, _ = env.reset()

    gray_sqr = "\U00002B1C"
    green_sqr = "\U0001F7E9"
    
    env_info = {
		'FrozenLake-v1': {
			'actions': lambda a: ["←","↓","→","↑"][a],
			'state_interpreter': lambda s: str(s)
		},
        'CartPole-v1': {
            'actions': lambda a: ["←","→"][a],
            'state_interpreter': lambda s: f"Position = {s[0]:.3f} m\nAngle = {s[2]*180/np.pi:.1f}°"
        },
        'LunarLander-v3': {
            'actions': lambda a: ["□□□", "■□□", "□■□", "□□■"][a],
            'state_interpreter': lambda s: (
                f"(x,y) = ({s[0]:.1f} m, {s[1]:.1f} m)\n"
                f"Leg Contact = ({bool(s[7])}, {bool(s[6])})"
            )
        },
        'MountainCar-v0': {
            'actions': lambda a: ["←","X","→"][a],
            'state_interpreter': lambda s: f"({s[0]:.2f}m, {s[1]:.2f}ms⁻¹)"
        },
        'Acrobot-v1': {
            'actions': lambda a: ["←","X","→"][a],
            'state_interpreter': lambda s: f"({np.arccos(s[0]):.2f}°, {np.arccos(s[1]):.2f}°)" 
        },
        'BipedalWalker-v3': {
            'actions': lambda a: np.round(np.asarray(a),2),
            'state_interpreter': lambda s: f"",
        }
    }

    step = 0
    
    while True:
        start_time = time.time()
        if is_discrete: 
            probs = policy(torch.FloatTensor(s))
            dist =  torch.distributions.Categorical(torch.as_tensor(probs))
            action = dist.sample().item()
        else: 
            mu, std = policy(torch.FloatTensor(s))
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            
        step += 1

        clear_output(wait=True)
        frame = env.render()

        plt.figure(figsize=figure_size)
        plt.imshow(frame)
        plt.axis('off')

        if env_id in env_info:
            interp = env_info[env_id]['state_interpreter'](s)
            action_str = env_info[env_id]['actions'](action)
        else:
            interp = ""
            action_str = str(action)

        # Add information below the image
        plt.text(0.5, -0.15, f"State: {interp}\nAction: {action_str}\nTime Step: {step}", 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='bottom', horizontalalignment='center')


        plt.show()

        s, r, terminated, truncated, _ = env.step(action.detach().numpy())
        r = float(r)
        
        end_time = time.time()
        if FPS:
            time.sleep(max(0,1 / FPS - (end_time - start_time)))
            
        if terminated or (truncated and do_truncate):
            break

    # Show final frame
    clear_output(wait=True)
    frame = env.render()

    plt.figure(figsize=figure_size)
    plt.imshow(frame)
    plt.axis('off')

    if env_id in env_info:
        interp = env_info[env_id]['state_interpreter'](s)
    else:
        interp = ""

    plt.text(0.5, -0.15, f"Final State: {interp}\nAction: {action_str}\nTime Step: {step}", 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='bottom', horizontalalignment='center')


    plt.show()


def plot_stats(stats, plots_info, window_size = 25):
	def moving_average(x, window_size=25):
		return np.convolve(x, np.ones(window_size)/window_size, mode='valid')
		
	subplot_shape = plots_info['subplt_shape']
	fig, axs = plt.subplots(*subplot_shape, figsize = plots_info['figsize'])
	
	ax_flat = axs.ravel()
	
	for i, info in enumerate(plots_info['subplts_info']):
		key = info['key']
		arr = stats[key]
		smoothed_arr = moving_average(arr, window_size)
		
		ax = ax_flat[i]
		ax.plot(np.arange(1, len(smoothed_arr) + 1), smoothed_arr)
		ax.set_title(info['title'])
		ax.set_xlabel(info['xlabel'])
		ax.set_ylabel(info['ylabel'])
		
	plt.tight_layout()
	plt.show()