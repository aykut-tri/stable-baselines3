import time
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control import utils

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


class MyEnv(gym.Env):
    """My custom environment derived from the Gym environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None):
        # Model parameters.
        self.dt = 1e-1
        self.nu = 2
        self.nq = 2
        self.nv = 2
        self.nx = self.nq + self.nv
        self.m = 0.1
        self.g = np.array([0, -9.8], dtype=np.float32)
        self.state = None

        # Normalized action limits.
        self.a_lb = -1e0*np.ones(self.nu, dtype=np.float32)
        self.a_ub = +1e0*np.ones(self.nu, dtype=np.float32)
        # Define the action space to be the input force.
        self.action_space = spaces.Box(low=self.a_lb,
                                       high=self.a_ub, dtype=np.float32)
        # Normalized action to force scale.
        self.a_to_f_scale = 1e1

        # State limits.
        self.q_lb = -1e0*np.ones(self.nq, dtype=np.float32)
        self.q_ub = 1e0*np.ones(self.nq, dtype=np.float32)
        self.v_lb = -1e2*np.ones(self.nv, dtype=np.float32)
        self.v_ub = 1e2*np.ones(self.nv, dtype=np.float32)
        self.x_lb = np.concatenate((self.q_lb, self.v_lb))
        self.x_ub = np.concatenate((self.q_ub, self.v_ub))
        # Define the observation space to be the state.
        self.observation_space = spaces.Box(
            low=self.x_lb, high=self.x_ub, dtype=np.float32)

        # The number of steps per epoch.
        self.num_epoch_steps = 0
        self.max_epoch_steps = 100

        # Rendering parameters.
        self.render_mode = render_mode
        self.screen = None
        self.screen_w = 600
        self.screen_h = 600
        self.clock = None
        self.isopen = True

    def step(self, action):
        # Check whether reset was called first.
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        # Count the number of steps in the epoch.
        self.num_epoch_steps += 1

        # Check for action feasibility.
        tol = 1e-6
        assert (np.all(action > self.a_lb - tol) and np.all(action < self.a_ub + tol)
                ), f"Normalized action {action} violates the limits"

        # Scale the action.
        force = action * self.a_to_f_scale

        # Get the current state.
        q = self.state[:self.nq]
        q_dot = self.state[self.nq:]

        # Evaluate the forward dynamics.
        q_ddot = force / self.m + self.g

        # Integrate via forward Euler.
        q = q + q_dot * self.dt
        q_dot = q_dot + q_ddot * self.dt

        # Return the new state.
        self.state = np.concatenate((q, q_dot))

        # Check for termination conditions.
        terminated = bool(
            (q < self.x_lb[:self.nq]).any() or
            (q > self.x_ub[:self.nq]).any() or
            (q_dot < self.x_lb[self.nq:]).any() or
            (q_dot > self.x_ub[self.nq:]).any() or
            (self.num_epoch_steps > self.max_epoch_steps)
        )

        # Calculate the reward.
        if not terminated:
            reward = 0.0

            # Penalize the deviation from a goal pose.
            goal = np.array([0.0, 0.0])
            distance_to_goal = np.linalg.norm(goal - q)
            task_cost = distance_to_goal * distance_to_goal

            # Penalize the energy of the system.
            energy_cost = q_dot.T @ q_dot
            # Penalize the effort.
            effort_cost = action.T @ action
            # Weigh and sum the cost terms.
            # if distance_to_goal <= 1e-1:
            #     reward += 1
            reward += np.exp(-5e0*task_cost)
            reward += 1e-1*np.exp(-energy_cost)
            reward += 0e-3*np.exp(-effort_cost)
        else:
            reward = -1.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        # Reset the epoch.
        self.num_epoch_steps = 0
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.9, 0.9)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        import pygame
        from pygame import gfxdraw

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_w, self.screen_h))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.screen_w, self.screen_h))
        surf.fill((50, 50, 50))

        scale_x = self.screen_w / (self.q_ub[0] - self.q_lb[0])
        scale_y = self.screen_h / (self.q_ub[1] - self.q_lb[1])

        img_x = self.state[0] * scale_x + self.screen_w / 2.0
        img_y = self.state[1] * scale_y + self.screen_h / 2.0
        ball_r = 0.05

        gfxdraw.filled_circle(surf, int(img_x), int(
            img_y), int(ball_r*scale_x), (0, 255, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

# Register the custom environment.
gym.envs.register(id="Ball2d-v0",
                  entry_point="test_custom_env:MyEnv")  # noqa

# Set the flags.
TRAIN = True

# Train/test.
if __name__ == "__main__":
    # Train the policy if opted.
    if TRAIN:
        # Create a vectorized environment to enable parallelization.
        vec_env = make_vec_env("Ball2d-v0", n_envs=32, seed=0,
                            vec_env_cls=SubprocVecEnv)
        # Train a model.
        print("\nTraining the policy...")
        model = PPO("MlpPolicy", vec_env, verbose=1,
                    tensorboard_log="./ball2d_tensorboard/",
                    n_steps=200)
        model.learn(total_timesteps=10000, tb_log_name="training")

        # Save the model.
        policy_file = "ball_2d_policy"
        print(f"Saving the policy into {policy_file}.zip")
        model.save(policy_file)
    # Load the policy otherwise.
    else:
        # Create an instance of the custom environment.
        env = gym.make("Ball2d-v0", render_mode="human")
        model = PPO.load("ball_2d_policy", env=env)

    # Evaluate the policy for a number of steps.
    vec_env = make_vec_env("Ball2d-v0", n_envs=1, seed=0,
                           vec_env_cls=SubprocVecEnv)
    NRUNS = 5
    for run_id in range(5):
        input(f"\n\nPress enter to start Execution {run_id+1}/{NRUNS}")
        obs = vec_env.reset()
        for i in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            print(f"\ti: {i}, a: {action}, q: {obs[0]}")
            vec_env.render("human")
            time.sleep(0.1)
