import time


class EnvironmentLoop:
    def __init__(self, environment, actor, should_update: bool = True):
        self._environment = environment
        self._actor = actor
        self._should_update = should_update

    def run_episode(self, num_steps):
        # Reset any counts and start the environment.
        start_time = time.time()
        episode_steps = 0

        # For evaluation, this keeps track of the total undiscounted reward
        # accumulated during the episode.
        episode_return = 0.0
        reward_list = []

        timestep = self._environment.reset()

        # Make the first observation.
        self._actor.observe_first(timestep)

        # Run an episode.
        while not timestep.last():
            if episode_steps >= num_steps:
                break

            # Generate an action from the agent's policy and step the environment.
            action = self._actor.select_action(timestep.observation)
            timestep = self._environment.step(action)

            reward_list.append(timestep.reward)

            # Have the agent observe the timestep and let the actor update itself.
            self._actor.observe(action, next_timestep=timestep)
            if self._should_update:
                self._actor.update()

            # Book-keeping.
            episode_steps += 1
            episode_return += timestep.reward
            if episode_steps % 100 == 0:
                print('episode_steps: ', episode_steps)    #PRINT STATEMENT

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
        }

        return result, reward_list
