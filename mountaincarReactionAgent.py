import gym 
#this is my reflex agent it works alright next step is looking at training an actual model
#observation space
    #0 car position Min -1.2 max .6
    #1 car velocity min -.07 max .07
#action space
    #0 accelerate left
    #1 dont accelerate
    #2 accelerate right
averageT=0
totalT=0
env = gym.make('MountainCar-v0') 
for i_episode in range(20): 
    observation = env.reset() 
    
    for t in range(200): 
        env.render() 
        
        
        if observation[1]<0:
            action=0
            if observation[0]<-.9:
                action=2
        else:
            action=2
        if t<3:
            action=2
        observation, reward, done, info = env.step(action) 
        print(t, observation, reward, done, info, action)
        if done: 
            print("Episode finished after {} timesteps".format(t+1)) 
            totalT+=t
            averageT=totalT/(i_episode+1)
            print("average Time ", averageT)
            break 


env.close()