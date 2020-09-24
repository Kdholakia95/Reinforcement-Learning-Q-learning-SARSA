import gym
import random
from gym.envs.registration import register
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

  def __init__(self):
        self.height = 12
        self.width = 12
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.height), spaces.Discrete(self.width)))
        self.moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.goal = None      # Will be set by learning algorithm
        self.viewer = None
        self.reset()
    
  def step(self, action):
        
        i = random.randint(1,10)
        
        if i == 1:
            i2 = random.randint(1,3)
            assigned_action = []            
            for i3 in range(4):
                if action != i3:
                    assigned_action.append(i3)                    
            if i2 == 1:
                action = assigned_action[0]
            elif i2 == 2:
                action = assigned_action[1]
            else:
                action = assigned_action[2]                    

        # State update due to action taken
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        # Westerly wind action
        if self.goal != (6,7):            # Disable wind for goal C            
            i4 = random.randint(1,2)
            if i4 == 1:
                self.S = self.S[0], self.S[1] + 1

        # Off-the-grid state
        self.S = max(0, self.S[0]), max(0, self.S[1])
        self.S = (min(self.S[0], self.height - 1), min(self.S[1], self.width - 1))                

        # Reward conditions
        if self.S == self.goal:	
            return self.S, 10, True, {}		  
        elif self.S[1] == 3:
            if self.S[0] in (2, 3, 4, 5, 6, 7, 8):
                return self.S, -1, False, {}
        elif self.S[1] == 8:
            if self.S[0] in (2, 3, 4, 5, 6):
                return self.S, -1, False, {}
        elif self.S[1] == 4:
            if self.S[0] in (2,8):
                return self.S, -1, False, {}
            elif self.S[0] in (3, 4, 5, 6, 7):
                return self.S, -2, False, {}
        elif self.S[1] == 7:
            if self.S[0] in (2, 6, 7, 8):
                return self.S, -1, False, {}
            elif self.S[0] in (3, 4, 5):
                return self.S, -2, False, {}
        elif self.S[1] == 5:
            if self.S[0] in (2, 8):
                return self.S, -1, False, {}
            elif self.S[0] in (3, 7):
                return self.S, -2, False, {}
            elif self.S[0] in (4, 5, 6):
                return self.S, -3, False, {}
        elif self.S[1] == 6:
            if self.S[0] in (2, 8):
                return self.S, -1, False, {}
            elif self.S[0] in (3, 5, 6, 7):
                return self.S, -2, False, {}
            elif self.S[0] == 4:
                return self.S, -3, False, {}                
        
        return self.S, 0, False, {}

  def reset(self):
        i = random.randint(1, 4)
        if i == 1:
            self.S = (5, 0)
        elif i == 2:        
            self.S = (6, 0)
        elif i == 3:
            self.S = (10, 0)
        else:
            self.S = (11, 0)
        return self.S
    
  def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            origin = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            trans = rendering.Transform(translation=(0, 0))
            
            agent.add_attr(trans)
                
            self.trans = trans
            agent.set_color(1, 0, 0)
            origin.set_color(0, 0, 0)
            origin.add_attr(rendering.Transform(
                translation=(screen_width // 2, screen_height // 2)))
            self.viewer.add_geom(agent)
            self.viewer.add_geom(origin)
            grid = make_polygon([(x+l,y+l),(x+l,y+r),(x+r,y+r),(x+r,y)])
            self.trans.set_translation(0, 0)
            self.trans.set_translation(
                (self.state[0] + 1) / 2 * screen_width,
                (self.state[1] + 1) / 2 * screen_height,
            )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
  #def close(self):

register(
    id='foo-v0',
    entry_point='gym_foo.foo_env:FooEnv',
)
