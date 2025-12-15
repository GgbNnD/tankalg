import multiprocessing as mp
import numpy as np
import os
from ai_controller import NeuralNetwork

# Worker function to run in a separate process
def worker(remote, parent_remote, env_fn_wrapper, opponent_weights=None):
    parent_remote.close()
    
    # Create environment
    # We need to suppress pygame output in workers
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    env = env_fn_wrapper()
    
    # Initialize opponent AI if weights are provided
    opponent_ai = None
    if opponent_weights is not None:
        # We need to know the input/hidden/output sizes
        # Inferring from weights
        input_size = opponent_weights['w1'].shape[0]
        hidden_size1 = opponent_weights['w1'].shape[1]
        
        if 'w3' in opponent_weights:
            # New model (2 hidden layers)
            hidden_size2 = opponent_weights['w2'].shape[1]
            output_size = opponent_weights['w3'].shape[1]
        else:
            # Old model (1 hidden layer)
            # Map old hidden size to both hidden1 and hidden2 to satisfy shape requirements for adaptation
            hidden_size2 = hidden_size1
            output_size = opponent_weights['w2'].shape[1]
        
        opponent_ai = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
        opponent_ai.set_weights(opponent_weights)
    
    try:
        # Keep track of opponent's observation
        # Initial reset
        # We don't reset here, we wait for 'reset' command or 'step' command?
        # Actually the main process calls reset() first.
        # But we need the observation for the opponent.
        # Let's initialize it as None and update it on reset/step.
        opponent_obs = None
        
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action_ai = data
                
                # Determine opponent action
                if opponent_ai is not None and opponent_obs is not None:
                    action_opponent = opponent_ai.forward(opponent_obs)
                else:
                    # Generate random opponent action
                    # [move, turn, shoot]
                    action_opponent = [
                        np.random.uniform(-1, 1), 
                        np.random.uniform(-1, 1), 
                        np.random.uniform(0, 1)
                    ]
                
                # Step environment
                # env.step returns: states, rewards, done, infos
                # states is list of 2 arrays (one for each player)
                # rewards is list of 2 floats
                next_states, rewards, done, infos = env.step([action_ai, action_opponent])
                
                # Update opponent observation
                opponent_obs = next_states[1]
                
                # Auto-reset if done
                if done:
                    reset_states = env.reset()
                    opponent_obs = reset_states[1]
                    # Return the reset state for the AI, but mark done=True so the trainer knows
                    remote.send((reset_states[0], rewards[0], True, infos[0]))
                else:
                    remote.send((next_states[0], rewards[0], False, infos[0]))
                    
            elif cmd == 'reset':
                states = env.reset()
                opponent_obs = states[1]
                remote.send(states[0])
                
            elif cmd == 'close':
                remote.close()
                break
                
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
                
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        pass

class SubprocVecEnv:
    def __init__(self, env_fns, opponent_weights=None):
        """
        env_fns: list of functions that return an env
        opponent_weights: dict of weights for the opponent AI (optional)
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Use spawn to be safe with CUDA/Torch if needed, though workers don't use torch here.
        # Fork is faster for startup but 'spawn' is safer.
        # Since workers only run numpy/pygame, 'fork' might be fine if we don't initialize torch in main before forking?
        # But we WILL initialize torch in main. So we MUST use 'spawn' or 'forkserver'.
        ctx = mp.get_context('spawn')
        
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, env_fn, opponent_weights))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
