import numpy as np
import random,pickle

class agent:
  def __init__(self,num_states_limit,num_actions,epsilon,gamma,lr,decay_rate):
    self.num_states_limit=num_states_limit
    self.num_actions=num_actions
    self.q=np.random.random((*self.num_states_limit,self.num_actions))
    self.epsilon=epsilon
    self.max_eps_val=epsilon
    self.decay_rate=decay_rate
    self.gamma=gamma
    self.lr=lr
    self.trainable=True
  
  def freeze(self):
    self.trainable=False

  def save_model_dict(self,path):
    with open(path,'wb') as f:
      pickle.dump(self.q,f)
  
  def load_model_dict(self,path):
    with open(path,'rb') as f:
      self.q=pickle.load(f)

  def train_one_step(self,last_state,last_action,new_state,reward):
    if self.trainable:
      tup=tuple(last_state)+(last_action,)
      if new_state is None:
        self.q[tup]=(1-self.lr)*self.q[tup]+self.lr*(reward) # v(next_state)=0
      else:
        self.q[tup]=(1-self.lr)*self.q[tup]+self.lr*(reward+self.gamma*np.max(self.q[new_state]))
  
  
  def do_action(self,state_encoding,eval_mode=False):
    if eval_mode or self.epsilon<random.random():#exploit
      return agent.argmax(self.q[tuple(state_encoding)])
    else:
      return random.randint(0,self.num_actions-1)
  
  def decay_exploration(self,t):
    self.epsilon=self.max_eps_val*np.exp(-self.decay_rate*t)

  @staticmethod
  def argmax(arr):
    maxi=np.max(arr)
    indices=[]
    for i in range(len(arr)):
      if arr[i]==maxi:
        indices.append(i)
    
    return random.choice(indices)