import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import matplotlib.pyplot as plt
import abc

class Bot:
    """
    Represents a bot with random features and reward.
    """
    def __init__(self, id, n_features):
        self.id = id
        self.n_features = n_features
        self.features = np.random.uniform(low=0.0, high=1.0, size = n_features)
        self.reward = np.random.randint(0, 10)

class Location:
    """
    Represents a location with a bot and location noise.
    """
    def __init__(self, id):
        self.id = id
        self.bot = 0
        self.location_noise = np.zeros(3)
        self.location_constant = 0


class LocationEnvironment3(gym.Env):
    """
    Represents an environment with multiple locations and bots.
    """
    def __init__(self, n_locations=4, n_bots=4, n_features=2, shuffle_interval=10000, episode_length=4):

        self.n_locations = n_locations
        self.n_bots = n_bots
        self.n_features = n_features
        self.shuffle_interval = shuffle_interval
        self.episode_length = episode_length
        self.action_space = spaces.Discrete(n_locations)  # an action chooses a location
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_locations, n_features), dtype=np.float32)
        self.bots = np.array([Bot(i, n_features) for i in range(n_bots)])
        self.locations = None
        self.timestep = 0
        self.mod_factor  = 2
        self.reset_day()
    
 
    def _get_obs(self):
        """
        Returns an observation of all the locations.
        """
        # obs = np.zeros((self.n_locations, self.n_features), dtype=float)
        bot_features = np.array([bot.features for bot in self.bots])  # shape (n_bots, n_features)
        bot_ids = np.array([location.bot for location in self.locations])  # shape (n_locations,)
        obs = np.take(bot_features, bot_ids, axis=0)  # shape (n_locations, n_features)
        location_masks = np.array([location.location_noise for location in self.locations])
        

        for loc in range(self.n_locations):
            for f in range(self.n_features):
                if location_masks[loc][f]==0:
                    obs[loc][f] = (obs[loc][f]+self.locations[loc].location_constant)%self.mod_factor


        return obs

    def _shuffle_bots(self):
        """
        Changes the location of the bots, without changing anything else.
        """
        random_bots = np.random.choice(np.arange(self.n_bots), size=self.n_locations, replace=False)
        for i, location in enumerate(self.locations):
            location.bot = random_bots[i]
    

    def reset_day(self,location_constant = 0):
        # every day not only location constant changes , but also the noise 
        self.locations = np.array([Location(i) for i in range(self.n_locations)])

        self.bots[0].features = np.array([1,1])
        self.bots[0].reward = 1
        self.bots[1].features = np.array([1,2])
        self.bots[1].reward = 0
        self.bots[2].features = np.array([2,1])
        self.bots[2].reward = 0
        self.bots[3].features = np.array([2,2])
        self.bots[3].reward = 0
        
        self.locations[0].location_noise = np.array([1,1])
        self.locations[1].location_noise = np.array([1,1])
        self.locations[2].location_noise = np.array([1,1])
        self.locations[3].location_noise = np.array([1,1])

        for i in range(self.n_locations):
            self.locations[i].location_constant = location_constant

        #self._shuffle_bots()

    # define reset hours func , where new bots are introduced
    def reset_hours():
        #todo
        a = 1
        
    def reset(self):
     
        self._shuffle_bots()
        self.timestep = 0
        return self._get_obs()

    def step(self, action):
        reward = self.bots[self.locations[action].bot].reward
        self.timestep += 1
        done = (self.timestep >= self.episode_length)
        return self._get_obs(), reward, done, {}
    

