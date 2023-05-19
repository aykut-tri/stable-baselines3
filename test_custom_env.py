import time
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control import utils

from stable_baselines3 import A2C, PPO, SAC


class MyEnv(gym.Env):
    """My custom environment derived from the Gym environment."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self):
        # Model parameters.
        self.dt = 1e-1
        self.nu = 2
        self.nq = 2
        self.nv = 2
        self.nx = self.nq + self.nv
        self.m = 0.1
        self.g = np.array([0, 0*-9.8], dtype=np.float32)
        self.state = None

        # Force limits.
        self.a_lb = -1e1*np.ones(self.nu, dtype=np.float32)
        self.a_ub = 1e1*np.ones(self.nu, dtype=np.float32)
        # Define the action space to be the input force.
        self.action_space = spaces.Box(low=self.a_lb,
                                       high=self.a_ub, dtype=np.float32)

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

        # Rendering parameters.
        self.render_mode = "human"
        self.screen = None
        self.screen_w = 600
        self.screen_h = 600
        self.clock = None
        self.isopen = True
        self.steps_beyond_terminated = None

    def step(self, action):
        # Check whether reset was called first.
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        # Check for action feasibility.
        assert (np.all(action > self.a_lb) and np.all(action < self.a_ub))

        # Get the current state.
        q = self.state[:self.nq]
        q_dot = self.state[self.nq:]

        # Evaluate the forward dynamics.
        q_ddot = action / self.m + self.g

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
            (q_dot > self.x_ub[self.nq:]).any()
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
            if distance_to_goal <= 1e-1:
                reward += 1
            reward -= 1e-1*energy_cost
            reward -= 0e-3*effort_cost
        else:
            reward = -1

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

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
        surf.fill((255, 255, 255))

        scale_x = self.screen_w / (self.q_ub[0] - self.q_lb[0])
        scale_y = self.screen_h / (self.q_ub[1] - self.q_lb[1])

        img_x = self.state[0] * scale_x + self.screen_w / 2.0
        img_y = self.state[1] * scale_y + self.screen_h / 2.0
        ball_r = 0.05

        gfxdraw.filled_circle(surf, int(img_x), int(
            img_y), int(ball_r*scale_x), (0, 255, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# Create an instance of the custom environment.
env = MyEnv()

# Visualize the environment for random states/actions.
print("\nVisualizing the environment for random states/actions...")
for i in range(5):
    env.reset()
    action = env.action_space.sample()
    env.step(action)
    env.render()
    print(f"\ti: {i}, action: {action.T}")
    time.sleep(1)

# Train the policy.
print("\nTraining the policy...")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# Evaluate the policy for a number of steps.
vec_env = model.get_env()
obs = vec_env.reset()
input("\n\nPress enter to test the policy")
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
