import multiprocessing as mp
import numpy as np
import os

# Worker function to run in a separate process
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    
    # Create environment
    # We need to suppress pygame output in workers
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    env = env_fn_wrapper()
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action_ai = data
                # Generate random opponent action here to save bandwidth
                # [move, turn, shoot]
                action_random = [
                    np.random.uniform(-1, 1), 
                    np.random.uniform(-1, 1), 
                    np.random.uniform(0, 1)
                ]
                
                # Step environment
                # env.step returns: states, rewards, done, infos
                # states is list of 2 arrays (one for each player)
                # rewards is list of 2 floats
                next_states, rewards, done, infos = env.step([action_ai, action_random])
                
                # Auto-reset if done
                if done:
                    reset_states = env.reset()
                    # Return the reset state for the AI, but mark done=True so the trainer knows
                    remote.send((reset_states[0], rewards[0], True, infos[0]))
                else:
                    remote.send((next_states[0], rewards[0], False, infos[0]))
                    
            elif cmd == 'reset':
                states = env.reset()
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
    def __init__(self, env_fns):
        """
        env_fns: list of functions that return an env
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
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, env_fn))
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
