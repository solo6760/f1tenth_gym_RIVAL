from gym.envs.registration import register
register(
	id='f110-v0',
	entry_point='f110_gym.envs:F110Env',
	# Retain the original unwrapped environment and its four-value reset/step.
	order_enforce=False,
	disable_env_checker=True,
	)