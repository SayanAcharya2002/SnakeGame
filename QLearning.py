import numpy as np
import random,pickle
import torch

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

    print(self.q.shape)

  def __str__(self):
    return f"""
q_shape: {self.q.shape}
num_actions:{self.num_actions}
eps:{self.epsilon}
lr:{self.lr}
gamma:{self.gamma}
trainable:{self.trainable}
        """
  
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
  

class CNN(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1=torch.nn.Conv2d(1,5,(2,2),padding="same") # 1,8,8->5,8,8
    self.act1=torch.nn.ReLU()
    self.maxpool1=torch.nn.MaxPool2d((2,2),stride=(1,1),padding=(1,1)) # 5,8,8->5,9,9

    self.conv2=torch.nn.Conv2d(5,15,(3,3),padding="valid") # 5,9,9->15,7,7
    self.act2=torch.nn.ReLU()
    self.maxpool2=torch.nn.MaxPool2d((3,3),stride=(1,1),padding=(0,0)) # 15,7,7->15,5,5

    self.conv3=torch.nn.Conv2d(15,64,(4,4),padding="valid") # 15,5,5->64,2,2
    self.act3=torch.nn.ReLU()
    self.maxpool3=torch.nn.MaxPool2d((2,2),stride=(1,1),padding=(0,0)) # 64,2,2->64,1,1

  def forward(self,x):

    x=self.conv1(x)
    x=self.act1(x)
    x=self.maxpool1(x)

    x=self.conv2(x)
    x=self.act2(x)
    x=self.maxpool2(x)

    x=self.conv3(x)
    x=self.act3(x)
    x=self.maxpool3(x)

    x=x.reshape(-1)
    return x

class Linear_ANN(torch.nn.Module):
  def __init__(self,in_size,out_size,hid_size=8):
    super().__init__()
    self.hidsize=hid_size

    self.lin=torch.nn.Sequential(
        torch.nn.Linear(in_features=in_size,out_features=4*self.hidsize),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=4*self.hidsize,out_features=2*self.hidsize),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=2*self.hidsize,out_features=self.hidsize),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=self.hidsize,out_features=out_size),
    )


  def forward(self,x):
    return self.lin(x)

class FuncApproxAgent(agent):
  ROTATION_MATRIX={
    #-90
    "left":np.array([
      [0,1],
      [-1,0]
    ]),
    #90
    "right":np.array([
      [0,-1],
      [1,0]
    ]),
    #180
    "down":np.array([
      [-1,0],
      [0,-1]
    ]),
    #0
    "up":np.array([
      [1,0],
      [0,1]
    ])
  }
  def __init__(self,flattened_pixel_limit,num_dir,num_actions,epsilon,gamma,lr,decay_rate,action_dir,height,width):
    self.flattened_pixel_limit=flattened_pixel_limit
    self.num_dir=num_dir
    self.num_actions=num_actions
    # input_size=>3*flattened_pixel_limit+num_dir
    # self.ann=Linear_ANN(3*flattened_pixel_limit+num_dir,num_actions)
    self.ann=Linear_ANN(0*flattened_pixel_limit+2,num_actions,4)
    self.cnn=CNN()

    self.epsilon=epsilon
    self.max_eps_val=epsilon
    self.decay_rate=decay_rate
    self.gamma=gamma
    self.lr=lr
    self.action_dir={action_dir[i]:i for i in range(len(action_dir))}
    self.height=height
    self.width=width

    self.optimizer=torch.optim.Adam(self.ann.parameters(),lr=1e-1)
    self.trainable=True

  def __str__(self):
    return f"""
            ann: {self.ann}
            cnn: {self.cnn}
            num_actions:{self.num_actions}
            eps:{self.epsilon}
            lr:{self.lr}
            gamma:{self.gamma}
            trainable:{self.trainable}
            """
  
  def freeze(self):
    self.trainable=False

  def save_model_dict(self,path):
    torch.save(self.ann,path+"_ANN")
    torch.save(self.cnn,path+"_CNN")
  
  def load_model_dict(self,path):
    self.ann=torch.load(path+"_ANN").cuda()
    self.cnn=torch.load(path+"_CNN").cuda()

  def get_state_encoding(self,state):

    return self.get_rotated_view(state)

    #snake body
    body=torch.zeros(self.flattened_pixel_limit)
    for x,y in state[0]:
      body[self.width*x+y]=1
    #snake head
    body[self.width*x+y]=0
    head=torch.zeros(self.flattened_pixel_limit)
    head[self.width*x+y]=1
    #fruit
    fruits=torch.zeros(self.flattened_pixel_limit)
    for fruit in state[1]:
      fruits[fruit[0]*self.width+fruit[1]]=1
    #direction
    direction=torch.zeros(self.num_dir)
    direction[self.action_dir[state[2]]]=1

    ans=torch.cat((head,self.cnn(body.reshape(1,self.height,self.width)),fruits,direction))
    
    return ans

  def get_rotated_view(self,state):
    aug_data=np.vstack((np.array(state[0]),np.array(state[1])))
    aug_data=np.transpose(aug_data)
    aug_data=FuncApproxAgent.ROTATION_MATRIX[state[2]]@aug_data #shape: (2,n)
    aug_data=np.where(aug_data<0,aug_data+np.array([self.width,self.height]).reshape(2,1),aug_data)
    body=aug_data[:,:-2]
    head=aug_data[:,-2]
    fruit=aug_data[:,-1]

    body_tensor=torch.zeros((self.width,self.height))
    head_tensor=torch.zeros_like(body_tensor)
    body_tensor[body[0],body[1]]=1
    head_tensor[head[0],head[1]]=1
    fruit_rel=torch.tensor(fruit-head)
    
    ans=torch.hstack((
      # head_tensor.reshape(-1),
      # (body_tensor.reshape(1,self.width,self.height)).reshape(-1),
      fruit_rel,
      )).to(dtype=torch.float32)

    return ans


  def train_one_step(self,last_state,last_action,new_state,reward):
    if self.trainable:
      self.optimizer.zero_grad()

      last_q=self.ann(self.get_state_encoding(last_state))
      if new_state is None:
        delta=(reward-last_q[last_action])
      else:
        new_q=self.ann(self.get_state_encoding(new_state))
        delta=(reward+self.gamma*((1-self.epsilon)*torch.max(new_q)+torch.sum(new_q)*self.epsilon/self.num_actions)-last_q[last_action])
      
      loss=-delta.detach()*last_q[last_action]
      loss.backward()

      self.optimizer.step()

  
  def do_action(self,state,eval_mode=False):
    if eval_mode or self.epsilon<random.random():#exploit
      
      return super().argmax(self.ann(self.get_state_encoding(state)).detach().cpu().numpy())
    else:
      return random.randint(0,self.num_actions-1)