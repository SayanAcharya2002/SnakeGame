import pygame
import numpy as np
from typing import Tuple,List
import itertools as it
import random,warnings


class BaseObject:
    def __init__(self,surface:pygame.Surface,parent_surface:pygame.Surface):
      self.surface=surface
      self.parent_surface=parent_surface

    def update_location(self):
      pass
  
    def clip_location(self):
      pass

    def get_rect(self):
      pass
      
    def render_object(self):
      raise NotImplementedError("Override the render_object function!!!")

    

class SnakePlayer(BaseObject):
  ACTIONS=["left","up","right","down"]
    
  def __init__(self,parent_surface:pygame.Surface,init_pos:Tuple[int,int],head_color:str,body_color:str,vel=10,block_size=10,grower=True):
    self.parent_surface=parent_surface
    self.head_color=head_color
    self.body_color=body_color
    self.vel=vel
    self.block_size=block_size
    self.is_alive=True
    self.grower=grower

    #head is the last element
    self.body:List[pygame.rect.Rect]=[pygame.rect.Rect(*init_pos,*(self.block_size,)*2)]
    #clockwise order (right arrow press)
    self.directions={
      "left":(-1,0), #left
      "up":(0,-1), # up
      "right":(1,0), # right
      "down":(0,1), # down
    }
    self.dir=random.choice(list(self.directions.keys()))
    self.body_set=set(part.topleft for part in self.body)

    self.autogrow=False # this flag controls whether the snake auto grows without apple
  
  def create_random_snake(self,_size):
    self.autogrow=True
    
    for _ in range(_size):
      rand_val=np.random.random()
      if rand_val<1/3:
        self.turn_left()
      elif rand_val<2/3:
        self.turn_right()
      
      self.update_location({})
  
    self.autogrow=False
    

  def __len__(self):
    return len(self.body)

  def turn_left(self):
    index=SnakePlayer.ACTIONS.index(self.dir)
    index=(index-1+len(SnakePlayer.ACTIONS))%len(SnakePlayer.ACTIONS)

    self.change_dir(SnakePlayer.ACTIONS[index])
  
  def turn_right(self):
    index=SnakePlayer.ACTIONS.index(self.dir)
    index=(index+1)%len(SnakePlayer.ACTIONS)

    self.change_dir(SnakePlayer.ACTIONS[index])    

  def is_colliding(self):
    # check head colliding with the map or the whole body
    map_width,map_height=self.parent_surface.get_size()
    
    head=self.body[-1]
    for part in self.body[:-1]:
      if head.colliderect(part):
        return True
    
    corner_x,corner_y=head.topleft
    if corner_x<0 or corner_y>map_height-self.block_size or corner_x>map_width-self.block_size or corner_y<0:
      # print("here:",corner_x,corner_y,corner_x+self.block_size,corner_y+self.block_size)
      return True
    
    return False

  def change_dir(self,move:str):
    prev_dir=self.dir
    if move not in self.directions:
      warnings.warn("Move not registered!!!")
    else:
      self.dir=move
      next_head=self.calculate_next_head()
      if next_head in self.body_set:
        self.dir=prev_dir
  
  def calculate_next_head(self)->Tuple[int,int]:
    head=self.body[-1]
    head=(head[0]+self.vel*self.directions[self.dir][0],head[1]+self.vel*self.directions[self.dir][1])
    return head

  def render_object(self):
    for part in self.body[:-1]:
      pygame.draw.rect(self.parent_surface,self.body_color,part)
    pygame.draw.rect(self.parent_surface,self.head_color,self.body[-1])

  def update_location(self,apples):
    head=self.calculate_next_head()
    self.body.append(pygame.rect.Rect(*head,*(self.block_size,)*2))
    self.body_set.add(head)
    if self.is_colliding(): # undo the changes
      self.is_alive=False
      self.body.pop()
      self.body_set.remove(head)

      return self.is_alive
    
    ate_apple=False
    for apple in apples:
      if self.body[-1].colliderect(pygame.rect.Rect(*apple)):
        ate_apple=True
        apples.remove(apple)
        break
    
    if not self.autogrow and (not self.grower or not ate_apple):
      #shift the body
      self.body_set.remove(self.body[0].topleft)
      self.body[:-1]=self.body[1::]
      self.body.pop()
  
    

    return self.is_alive

def get_image_surface(path,dim):
    image_surface=pygame.image.load(path)
    image_surface=pygame.transform.scale(image_surface,dim)
    return image_surface