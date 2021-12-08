import numpy as np
import gym
import matplotlib.pyplot as plt
#this is the link to the website that gave me the inspiration for this
#https://towardsdatascience.com/getting-started-with-reinforcement-learning-and-open-ai-gym-c289aca874f

env = gym.make('MountainCar-v0')
env.reset()


def rienforcementLearning(env, learning, discount, epsilon, minEpisodes, episodes):
    
    numStates = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    numStates = np.round(numStates, 0).astype(int) + 1
    
    Q = np.random.uniform(low = -1, high = 1, size = (numStates[0], numStates[1], env.action_space.n))
    
    rewardList = []
    averageRewardList = []
    
    reduction = (epsilon - minEpisodes)/episodes
    
    for i in range(episodes):
        done = False
        totReward, reward = 0,0
        state = env.reset()
        
        stateAdj = (state - env.observation_space.low)*np.array([10, 100])
        stateAdj = np.round(stateAdj, 0).astype(int)
    
        while done != True:
            if i >= (episodes - 20):
                env.render()
            
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[stateAdj[0], stateAdj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
            
            state2, reward, done, info = env.step(action) 
            
            
            state2Adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2Adj = np.round(state2Adj, 0).astype(int)
            
            
            if done and state2[0] >= 0.5:
                Q[stateAdj[0], stateAdj[1], action] = reward
                
            
            else:
                delta = learning*(reward + discount*np.max(Q[state2Adj[0], state2Adj[1]]) - Q[stateAdj[0], stateAdj[1],action])
                Q[stateAdj[0], stateAdj[1],action] += delta
                                     
            
            totReward += reward
            stateAdj = state2Adj
        
        if epsilon > minEpisodes:
            epsilon -= reduction
        
        rewardList.append(totReward)
        
        if (i+1) % 100 == 0:
            averageReward = np.mean(rewardList)
            averageRewardList.append(averageReward)
            rewardList = []
            
        if (i+1) % 100 == 0:    
            print('Episode {} Average Reward: {}'.format(i+1, averageReward))
            
    env.close()
    
    return averageRewardList

#rewards = rienforcementLearning(env, 0.2, 0.9, 0.8, 0, 10000)
rewards = rienforcementLearning(env, 0.02, .99, .8, 0, 5000)

plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')     
plt.close()  