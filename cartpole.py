!apt-get install -y xvfb x11-utils
!pip install pyvirtualdisplay==0.2.*

!pip install 'ray[rllib]'==1.6

import numpy as np
import time
import math

import gym
env = gym.make("CartPole-v1")

env.action_space
env.reset()

def random_movements():
    time = 0
    average_time = 0
    episodes = 20

    for episode in range(episodes):
        # resets environment before each loop
        env.reset()

        # sets length at 500
        for t in range(500):

            # commented render line out for the sake of saving time
            # env.render()
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            print(t, observation, reward, terminated, truncated, info, action)
            if terminated or truncated:
                # calculates average time the pole is balanced
                time = t
                average_time += time
                break
    average_time = average_time / episodes
    print("Average time balancing pole: ", average_time)

random_movements()


from pyvirtualdisplay import Display
display = Display(visible=False, size=(1400, 900))
_ = display.start()


from gym.wrappers.monitoring.video_recorder import VideoRecorder
before_training = "before_training.mp4"

video = VideoRecorder(env, before_training)
# returns an initial observation
env.reset()
for i in range(200):
  env.render()
  video.capture_frame()
  # env.action_space.sample() produces either 0 (left) or 1 (right).
  observation, reward, done, info = env.step(env.action_space.sample())

video.close()
env.close()


from base64 import b64encode
def render_mp4(videopath: str) -> str:
  """
  Gets a string containing a b4-encoded version of the MP4 video
  at the specified path.
  """
  mp4 = open(videopath, 'rb').read()
  base64_encoded_mp4 = b64encode(mp4).decode()
  return f'<video width=400 controls><source src="data:video/mp4;' \
         f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'


from IPython.display import HTML
html = render_mp4(before_training)
HTML(html)



# This cell is not my own work but I wanted to include it as it was the option
# I wanted to work with, but didn't understand enough of to feel comfortable using

import ray
from ray.rllib.agents.ppo import PPOTrainer
config = {
    "env": "CartPole-v1",
    # Change the following line to `“framework”: “tf”` to use tensorflow
    "framework": "torch",
    "model": {
      "fcnet_hiddens": [32],
      "fcnet_activation": "linear",
    },
}
 stop = {"episode_reward_mean": 195}
 ray.shutdown()
ray.init(
  num_cpus=3,
  include_dashboard=False,
  ignore_reinit_error=True,
  log_to_driver=False,
)
# execute training 
analysis = ray.tune.run(
  "PPO",
  config=config,
  stop=stop,
  checkpoint_at_end=True,
)

# restore a trainer from the last checkpoint
trial = analysis.get_best_logdir("episode_reward_mean", "max")
checkpoint = analysis.get_best_checkpoint(
  trial,
  "training_iteration",
  "max",
)
trainer = PPOTrainer(config=config)
trainer.restore(checkpoint)



after_training = "after_training.mp4"
after_video = VideoRecorder(env, after_training)
observation = env.reset()
done = False

while not done:
  env.render()
  after_video.capture_frame()
  action = trainer.compute_action(observation)
  observation, reward, done, info = env.step(action)
after_video.close()
env.close()
 
html = render_mp4(after_training)
HTML(html)



# Q-table data
from tabulate import tabulate
q_table_data = [
    (0, [-0.04600373, 0.21646295, 0.01235373, -0.30213776], 1.0, False, {}),
    (1, [-0.04167448, 0.02116713, 0.00631098, -0.00558448], 1.0, False, {}),
    (2, [-0.04125113, 0.216198, 0.00619929, -0.29626957], 1.0, False, {}),
    (3, [-0.03692717, 0.02098823, 0.00027389, -0.00163794], 1.0, False, {}),
    (4, [-3.6507409e-02, -1.7413765e-01, 2.4113544e-04, 2.9113138e-01], 1.0, False, {}),
    (5, [-0.03999016, 0.02098086, 0.00606376, -0.00147548], 1.0, False, {}),
    (6, [-0.03957054, -0.17422752, 0.00603425, 0.29311445], 1.0, False, {}),
    (7, [-0.04305509, 0.02080787, 0.01189654, 0.00234074], 1.0, False, {}),
    (8, [-0.04263894, -0.17448266, 0.01194336, 0.2987533], 1.0, False, {}),
    (9, [-0.04612859, -0.3697728, 0.01791842, 0.5951789], 1.0, False, {}),
    (10, [-0.05352405, -0.5651409, 0.029822, 0.8934517], 1.0, False, {}),
    (11, [-0.06482686, -0.3704358, 0.04769104, 0.61029035], 1.0, False, {}),
    (12, [-0.07223558, -0.56619084, 0.05989684, 0.9176048], 1.0, False, {}),
    (13, [-0.08355939, -0.76206917, 0.07824894, 1.2284946], 1.0, False, {}),
    (14, [-0.09880078, -0.5680365, 0.10281883, 0.96131796], 1.0, False, {}),
    (15, [-0.11016151, -0.7643788, 0.12204519, 1.2844515], 1.0, False, {}),
    (16, [-0.12544909, -0.9608247, 0.14773422, 1.6127208], 1.0, False, {}),
    (17, [-0.14466558, -1.1573496, 0.17998864, 1.9475732], 1.0, False, {}),
    (18, [-0.16781257, -0.9645433, 0.2189401, 1.7156669], 1.0, True, {'TimeLimit.truncated': False}),
    (19, [-0.18710344, -1.1614025, 0.25325343, 2.0677621], 0.0, True, {'TimeLimit.truncated': False})
]

# Print out the Q-table
print(tabulate(q_table_data, headers=["Step", "Q-Values", "Reward", "Done", "Info"], tablefmt="fancy_grid"))



def get_discrete_state(state):
    discrete_state = state/np_array_win_size+ np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))

def q_learning():
    learning_rate = 0.1
    discount = 0.95
    episodes = 1000
    total = 0
    total_reward = 0
    prior_reward = 0

    np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

    epsilon = 1
    epsilon_decay = 0.99995

    for episode in range(EPISODES + 1):
      t0 = time.time() 
      discrete_state = get_discrete_state(env.reset()) 
      done = False
      episode_reward = 0 
