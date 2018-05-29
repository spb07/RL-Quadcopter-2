import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent. Goal is to takeoff to a given height  and hover once takeoff height is achieved. Ideally, only vertical movement with no movement in other planes and no rotation"""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 2 # multiplier is equal to space size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1
        #self.init_velocities = init_velocities
        #self.target_pos = target_pos

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
       #self.target_v = np.array([0., 0.])
       #self.target_angular_v = np.array([0., 0., 0.])
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        '''
        if (abs(self.sim.pose[2] - self.target_pos[2]))<0.3: #within 30cm of target height
            prize= 1
        else:
            if (self.sim.pose[2] > (2* self.target_pos[2])): # penalty for overshooting target height
                prize = -1
            else:
                if ((self.sim.pose[2] - self.target_pos[2])/self.sim.v[2])< 0: # Reward for going in right direction
                    prize=0.2
                else: # penalty for drifting away from target height
                    prize=-0.2
        '''            
            
        #Position based reward
        pos = (self.sim.pose[2]/self.target_pos[2]) #relative position of quadcopter to target height
        if pos > 3: #overshot target height by 3 times
            prize =-1 
        else:
            prize= np.sin(pos * (np.pi/2.))  #reward increases smoothly to 1 till target height and then decrease smootly to -1 when current height is 3 times target height, with an additional reward/penalty based on whether quad is going in right direction
         
        # Direction of travel reward
        if ((self.sim.pose[2] - self.target_pos[2])/self.sim.v[2])< 0: # Reward for going in right direction
            direc = 0.3
        else: # penalty for drifting away from target height
            direc = -0.3

        # Reward determination
        if self.sim.pose[2] <self.sim.init_pose[2]: #penalty for not going above initial position
            reward = -1
        else:
            if (abs(self.sim.v[2])>self.target_pos[2]/2): # penalty for excessive speed
                reward = -1
            else:
                if self.sim.done:
                    if self.sim.time < self.sim.runtime: #penalty for hitting boundary before runtime
                        reward = -1
                    else: # episode ran for full runtime
                        finish = 50/(1+(abs(self.sim.pose[2] - self.target_pos[2]))) #special reward for finishing episode, with maximum reward when finish position is at target height
                        reward = prize + direc + finish
                else: # continuous reward during episode
                    reward = prize + direc     
        
        
        '''
        if (abs(self.sim.pose[2] - self.target_pos[2]))<0.3: #within 30cm of target height
            prize= 5
        else:
            if (self.sim.pose[2] > (2* self.target_pos[2])): # penalty for overshooting target height
                prize = -5
            else:
                if ((self.sim.pose[2] - self.target_pos[2])/self.sim.v[2])< 0: # Reward for going in right direction
                    prize=1
                else: # penalty for drifting away from target height
                    prize=-1

        if self.sim.pose[2] <self.sim.init_pose[2]: #penalty for not going above initial position
            reward = -5
        else:
            if self.sim.done:
                if self.sim.time < self.sim.runtime: #penalty for hitting boundary before runtime
                    reward = -2
                else: # episode ran for full runtime
                    reward = prize
            else: # continuous reward during episode
                reward = prize
        '''        
        #reward = 1.- np.tanh(abs(self.sim.pose[2] - self.target_pos[2])) #only reward reaching the height
        #reward = 1.-.3*(abs(self.sim.pose[2] - self.target_pos[2])).sum() 
        #reward = self.sim.pose[2] #quad went to zero height from starting height of 10
        #reward = 1.-.3*(abs(self.sim.pose[2] - self.target_pos[2])).sum() #only reward reaching the height
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        #reward = np.tanh(3.-.9*(abs(self.sim.pose[:3] - self.target_pos)).sum()-.2*(abs(self.sim.v[:2] -self.target_v)).sum()-.2*(abs(self.sim.angular_v[:3] -self.target_angular_v)).sum())
        #print("\n Time= = {:7.3f} Z= {:7.3f} , VZ = {:7.3f} ,Accel= {:7.3f}, ,Prize= {:7.4f}, Direc= {:7.4f}, Reward= {:7.4f} ".format( self.sim.time, self.sim.pose[2],self.sim.v[2],self.sim.linear_accel[2],prize, direc, reward ), end="") 
        return reward

        
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(np.concatenate([rotor_speeds] * (4))) # updates pose, v and angular_v. Returns True if env bounds breached or time up
            reward += self.get_reward() 
            #pose_all.append(self.sim.pose)
            pose_all.append(np.concatenate(([self.sim.pose[2]],[self.sim.v[2]]),axis =0))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.takeoff= False 
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat) # state definition
        #print("Input init velocity reset mod: ", self.sim.init_velocities)
        #print("Input init position reset mod: ", self.sim.init_pose)
        #print("Target pos reset mod: ", self.target_pos)
        #print("Reset velocity in reset mod: ", self.sim.v)
        state = np.concatenate(([self.sim.pose[2]],[self.sim.v[2]])*self.action_repeat,axis =0)
        #state = np.concatenate([self.sim.pose[2] * self.action_repeat) #restrict to height only
        return state
    
 