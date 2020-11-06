from spinup import ppo_pytorch as ppo
import torch.nn as nn
import gym
import gym_panda
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import spinup.algos.pytorch.ppo.core as core

entrypoints = torch.hub.list('pytorch/vision:v0.5.0', force_reload=True)

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self,env=None):
        super(ProcessFrame84,self).__init__(env)
        self.env=env
        self.observation_space=gym.spaces.Box(
            low=0,high=255,shape=(84,84,1),dtype=np.uint8
        )

    def observation(self, observation):
        return ProcessFrame84.process(self.env.render(mode='rgb_array'))

    def process(frame):
        if frame.size==720*960*3:
            img=np.reshape(frame,[720,960,3]).astype(np.float32)
        else:
            assert False,"Unknown resolution."
        img=img[:,:,0]*0.399+img[:,:,1]*0.587+img[:,:,2]*0.114
        resized_screen=cv2.resize(img,(112,84),interpolation=cv2.INTER_AREA)
        y_t=resized_screen[:,14:98]
        y_t=np.reshape(y_t,[84,84,1])
        return y_t.astype(np.uint8)


class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self,env):
        super(ImageToPytorch,self).__init__(env)
        old_shape=self.observation_space.shape
        new_shape=(old_shape[-1],old_shape[0],old_shape[1])
        self.observation_space=gym.spaces.Box(
            low=0.0,high=1.0,shape=new_shape,dtype=np.float32
        )

    def observation(self, observation):
        return np.moveaxis(observation,2,0)


class MoveTowardZ(gym.ActionWrapper):
    def __init__(self,env):
        super(MoveTowardZ,self).__init__(env)

    def action(self,action):
        action[2]=-0.3
        return action


env=gym.make('panda-v0')
env=ProcessFrame84(env)
env=ImageToPytorch(env)
env=MoveTowardZ(env)


# image=env.reset()
# plt.figure()
# plt.imshow(image.squeeze(),cmap='gray')
# plt.title("example extracted screen")
# plt.show()

env_fn=lambda:env
ac_kwargs=dict(hidden_sizes=[18,64,64],activation=nn.ReLU)
logger_kwargs=dict(output_dir='spinup',exp_name='panda_ppo')
ppo(env_fn=env_fn,actor_critic=core.CNNActorCritic,ac_kwargs=ac_kwargs,steps_per_epoch=5000,epochs=250,logger_kwargs=logger_kwargs)