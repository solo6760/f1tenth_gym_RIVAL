"""Headless regression coverage for the legacy Gym four-tuple API."""

import copy
from pathlib import Path
import unittest

import gym
import numpy as np
import yaml


class CompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        examples = Path(__file__).resolve().parents[3] / 'examples'
        with (examples / 'config_example_map.yaml').open() as stream:
            cls.config = yaml.safe_load(stream)
        cls.map_path = str((examples / cls.config['map_path']).resolve())

    def make_env(self, num_agents):
        return gym.make(
            'f110_gym:f110-v0',
            map=self.map_path,
            map_ext=self.config['map_ext'],
            num_agents=num_agents,
            seed=self.config['seed'],
        )

    def poses(self, num_agents):
        pose = np.array([
            self.config['sx'], self.config['sy'], self.config['stheta']
        ], dtype=np.float64)
        poses = np.tile(pose, (num_agents, 1))
        # Start the second car behind the first without overlapping it.
        for index in range(1, num_agents):
            poses[index, :2] -= index * np.array([
                np.cos(pose[2]), np.sin(pose[2])
            ])
        return poses

    def assert_transition(self, result, num_agents):
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        obs, reward, done, info = result
        self.assertIsInstance(obs, dict)
        self.assertEqual(obs['ego_idx'], 0)
        scans = np.asarray(obs['scans'])
        self.assertEqual(scans.shape, (num_agents, 1080))
        self.assertTrue(np.isfinite(scans).all())
        for key in (
            'poses_x', 'poses_y', 'poses_theta', 'linear_vels_x',
            'linear_vels_y', 'ang_vels_z', 'collisions',
            'lap_times', 'lap_counts',
        ):
            with self.subTest(observation=key):
                values = np.asarray(obs[key])
                self.assertEqual(values.shape, (num_agents,))
                self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(reward, 0.01)
        self.assertIsInstance(done, (bool, np.bool_))
        self.assertIsInstance(info, dict)
        self.assertEqual(np.asarray(info['checkpoint_done']).shape, (num_agents,))

    def test_reset_and_step_four_tuple_api(self):
        for num_agents in (1, 2):
            with self.subTest(num_agents=num_agents):
                env = self.make_env(num_agents)
                try:
                    self.assertIs(env, env.unwrapped)
                    poses = self.poses(num_agents)
                    result = env.reset(poses)
                    self.assert_transition(result, num_agents)
                    for column, key in enumerate(('poses_x', 'poses_y', 'poses_theta')):
                        np.testing.assert_allclose(result[0][key], poses[:, column])
                    self.assertFalse(result[2])
                    actions = np.tile([0.05, 1.0], (num_agents, 1))
                    for _ in range(5):
                        result = env.step(actions)
                        self.assert_transition(result, num_agents)
                        self.assertFalse(result[2])
                    self.assertTrue(np.all(np.asarray(result[0]['linear_vels_x']) > 0))
                finally:
                    env.close()

    def rollout(self, num_agents):
        env = self.make_env(num_agents)
        try:
            # Snapshot mutable lap/collision arrays before subsequent steps.
            history = [copy.deepcopy(env.reset(poses=self.poses(num_agents)))]
            for steering in (0.0, 0.05, -0.05, 0.02, 0.0):
                actions = np.tile([steering, 1.0], (num_agents, 1))
                history.append(copy.deepcopy(env.step(actions)))
            return history
        finally:
            env.close()

    def test_seeded_separate_environment_rollouts(self):
        for num_agents in (1, 2):
            with self.subTest(num_agents=num_agents):
                # RaceCar shares its scan simulator: finish one rollout before
                # constructing the next environment, rather than interleaving.
                first = self.rollout(num_agents)
                second = self.rollout(num_agents)
                self.assertEqual(len(first), len(second))
                for index, (left, right) in enumerate(zip(first, second)):
                    with self.subTest(step=index):
                        self.assert_transition(left, num_agents)
                        self.assert_transition(right, num_agents)
                        self.assertEqual(left[0].keys(), right[0].keys())
                        for key in left[0]:
                            np.testing.assert_array_equal(
                                left[0][key], right[0][key], err_msg=key
                            )
                        self.assertEqual(left[1:3], right[1:3])
                        self.assertEqual(left[3].keys(), right[3].keys())
                        for key in left[3]:
                            np.testing.assert_array_equal(left[3][key], right[3][key])


if __name__ == '__main__':
    unittest.main()
