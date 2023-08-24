import pygame,random,os,time
import utils
import QLearning
import numpy as np
from typing import List
from tqdm import tqdm
from copy import deepcopy

pygame.mixer.init()
pygame.font.init()
FONT = pygame.font.SysFont('comicsans', 40)

COLLISION_EVENT=pygame.USEREVENT+1
FPS=10
IMAGE_ASSET_PATH = os.path.join(os.path.dirname(__file__),"assets","images")
SOUND_ASSET_PATH = os.path.join(os.path.dirname(__file__),"assets","sounds")
MODEL_DIR=os.path.join(os.path.dirname(__file__),"models")

WIDTH,HEIGHT=800,800

WIN=pygame.display.set_mode((WIDTH,HEIGHT))
BACKGROUND=utils.get_image_surface(os.path.join(IMAGE_ASSET_PATH,"bluish.jpg"),(WIDTH,HEIGHT))
PLAYER_SIZE=100

# PLAYER=utils.SnakePlayer(WIN,(WIDTH//2,HEIGHT//2),head_color="black",body_color="green",vel=PLAYER_SIZE,block_size=PLAYER_SIZE)
# apples:set[pygame.rect.Rect]=set()
is_render=True

NUM_DIR=4
ALMOST_INF=int(1e9)
ACTION_ORDERING=["left","up","right","down"]
BOUNDARY_VALS=[(0,ALMOST_INF),(ALMOST_INF,0),(WIDTH-PLAYER_SIZE,ALMOST_INF),(ALMOST_INF,HEIGHT-PLAYER_SIZE)]
BOUNDARY_VALS=[np.array(tup)//PLAYER_SIZE for tup in BOUNDARY_VALS]
KEY_ORDERING=[pygame.K_LEFT,pygame.K_RIGHT,pygame.K_UP,pygame.K_DOWN]
KEY_MAPPINGS={
  key:action for key,action in zip(KEY_ORDERING,ACTION_ORDERING)
}
DIR_MAPPING={
  action:enum for enum,action in enumerate(ACTION_ORDERING)
}

def get_state_encoding(player:utils.SnakePlayer,apples):
  """
    return a tuple of length 5 containing the following:
    [
      length,
      front_boundary_dist,
      right_boundary_dist,
      rel_apple_x,
      rel_apple_y,
    ]
  """
  head=np.array(player.body[-1].topleft)//PLAYER_SIZE
  # tail=np.array(player.body[0].topleft)//PLAYER_SIZE
  # mid=np.array(player.body[len(player.body)//2].topleft)//PLAYER_SIZE
  apple_dist=(np.array(list(apples)[0])//PLAYER_SIZE)[:2]
  dir=DIR_MAPPING[player.dir]
  
  state=[]
  state.append(len(player)-1) # always greater than 1
  state.append(np.min(np.abs(BOUNDARY_VALS[dir]-head)))
  state.append(np.min(np.abs(BOUNDARY_VALS[(dir+1)%NUM_DIR]-head)))
  state.extend(i for i in apple_dist-head+WIDTH//PLAYER_SIZE)
  return tuple(state)

def render_screen():
  if is_render:
    WIN.blit(BACKGROUND,(0,0))
    
  valid=PLAYER.update_location(apples)
  if not valid:
    pygame.event.post(pygame.event.Event(COLLISION_EVENT))
  else:
    if is_render:
      PLAYER.render_object()
  if is_render:
    for apple in apples:
      pygame.draw.rect(WIN,"red",pygame.rect.Rect(apple))
  if is_render:
    SHOW_TEXT=FONT.render(f"Length:{len(PLAYER)}",1,(255,0,0),(0,0,0))
    SHOW_TEXT.set_alpha(0.7*256)
    WIN.blit(SHOW_TEXT,(WIDTH-SHOW_TEXT.get_width(),0))
    pygame.display.update()

def render_end_screen(win):
  if win:
    SHOW_TEXT=FONT.render("won the game",1,(255,0,0),(0,0,0))
  else:
    SHOW_TEXT=FONT.render("lost the game",1,(255,0,0),(0,0,0))
  WIN.blit(SHOW_TEXT,(WIDTH//2-SHOW_TEXT.get_width()//2,HEIGHT//2-SHOW_TEXT.get_height()//2))
  pygame.display.update()

def generate_random_apple():
  return (random.randrange(0,WIDTH-PLAYER_SIZE+1,PLAYER_SIZE),\
                  random.randrange(0,HEIGHT-PLAYER_SIZE+1,PLAYER_SIZE))

def generate_random_snake():
  return (random.randrange(0,WIDTH-PLAYER_SIZE+1,PLAYER_SIZE),\
                  random.randrange(0,HEIGHT-PLAYER_SIZE+1,PLAYER_SIZE))

if __name__=="__main__":
  num_episode=1_000_000
  episode_factor=1000
  max_step_per_episode=1000
  agent_playing=True
  max_snake_length=6

  clock=pygame.time.Clock()
  best_model_path=os.path.join(MODEL_DIR,f"QLearningAgentBest_{WIDTH//PLAYER_SIZE}x{HEIGHT//PLAYER_SIZE}_{max_snake_length}.pkl")
  latest_model_path=os.path.join(MODEL_DIR,f"QLearningAgentLatest_{WIDTH//PLAYER_SIZE}x{HEIGHT//PLAYER_SIZE}_{max_snake_length}.pkl")


  is_render=True
  
  min_lr=0.01
  max_lr=0.1
  best_agg_reward=-np.inf
  eval_mode=False
  agent=QLearning.agent(
    num_states_limit=(max_snake_length+1,)+(WIDTH//PLAYER_SIZE,)*2+(2*WIDTH//PLAYER_SIZE+1,)*2,
    num_actions=3, # turn_left,turn_right, and noop
    epsilon=0.01,
    gamma=0.9,
    lr=max_lr,
    decay_rate=0,
  )



  if is_render:
    print("testing model")
    max_step_per_episode=10_000
    FPS=20
    eval_mode=False
    agent.load_model_dict(latest_model_path)
    agent.freeze()
  else:
    print("training model")
    FPS=100_000
    try:
      agent.load_model_dict(latest_model_path)
    except FileNotFoundError:
      pass
    
  
  agg_reward_list=[]
  pos_reward=[]
  for episode in range(num_episode):
    # print(episode,agent.epsilon)
    if not is_render:
      agent.lr=max_lr-(max_lr-min_lr)*episode/num_episode
      agent.decay_exploration(episode/num_episode)

    PLAYER=utils.SnakePlayer(WIN,generate_random_snake(),head_color="yellow",body_color="green",vel=PLAYER_SIZE,block_size=PLAYER_SIZE,grower=True)
    apples:set[pygame.rect.Rect]=set([(*generate_random_apple(),PLAYER_SIZE,PLAYER_SIZE)])
    run=True
    win=True
    rewards=[]
    
    cur_state=None
    index=0
    snake_length=1
    while run:
      index+=1
      if(index==max_step_per_episode):
        # print(f"played more than 1000 at episode:{episode}")
        agent.train_one_step(cur_state,action,None,-10)
        rewards.append(-10)
        pygame.event.clear()
        break
        
      clock.tick(FPS)
      for event in pygame.event.get():
        if event.type==pygame.QUIT:
          run=False
          break
        if event.type==COLLISION_EVENT:
          agent.train_one_step(cur_state,action,None,-1)
          rewards.append(-1)
          run=False
          win=False
          break

      if not run:
        break

      if cur_state is not None:
        #update agent
        agent.train_one_step(cur_state,action,get_state_encoding(PLAYER,apples),2*reward*snake_length)
        rewards.append(2*reward*snake_length)
      
      if snake_length>max_snake_length:
        break

      if agent_playing:
        cur_state=get_state_encoding(PLAYER,apples)
        def at_border(state):
          return state[1]==0 or state[2]==0
        action=agent.do_action(cur_state,not is_render or (eval_mode and not at_border(cur_state)))
        if action==2:
          pass # do nothing
        elif action==0:
          PLAYER.turn_left()
        else:
          PLAYER.turn_right()
      
      else:
        keys=pygame.key.get_pressed()
        for key,cmd in KEY_MAPPINGS.items():
          if keys[key]:
            PLAYER.change_dir(cmd)

      render_screen()

      reward=0
      if len(apples)==0:
        reward=1
        snake_length+=reward
        apple_loc=None
        while not apple_loc or apple_loc in PLAYER.body_set:
          apple_loc=generate_random_apple()
        apples.add((*apple_loc,PLAYER_SIZE,PLAYER_SIZE))
        # print(apple_loc)

      
    cur_agg_reward=np.sum(rewards)
    cur_pos_reward=np.sum(i for i in rewards if i>0)
    pos_reward.append(cur_pos_reward)
    agg_reward_list.append(cur_agg_reward)
    if not is_render:
      if episode%episode_factor==0:
        print(f"episode: {episode}")
        print(agent)
        print(f"agg_reward last {episode_factor}:{np.average(agg_reward_list[-1:-(episode_factor+1):-1])}")
        # print(f"pos: {pos} neg:{neg}")
        # print(f"agg. pos reward: {np.mean(pos_reward[-1:-(episode_factor+1):-1])}")
        print(f"saving latest model at episode:{episode}...............")
        agent.save_model_dict(latest_model_path)

      if cur_agg_reward>best_agg_reward:
        best_agg_reward=cur_agg_reward
        print(f"reward:{best_agg_reward} saving best model at episode:{episode}...............")
        # best_agent=deepcopy(agent)
        agent.save_model_dict(best_model_path)
    
  print(f"best_case: {best_agg_reward}")
  if is_render:
    render_end_screen(win)
    time.sleep(2)
    pygame.quit()
