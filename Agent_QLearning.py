import gym 
import itertools 
import matplotlib.pyplot as plt
import random
import numpy as np
from gym_foo import foo_env

    
def qlearning(env, episodes, runs):
    state_dim = env.height * env.width
    action_dim = 4

    # Q table and parameter tuning
    #Q = np.zeros([state_dim, action_dim])
    Q = np.random.rand(state_dim, action_dim)                                  
    
    Q[12*env.goal[0] + env.goal[1]][:] = 0
    gamma = 0.9
    alpha = 0.6 
    epsilon = 0.1  

    # Data for plotting
    x_time = []
    x_reward = np.zeros(episodes)    
    
    for ep in range(episodes): 
        
        # reset state
        state = env.reset()        
        
        for t in itertools.count():
            
            action = epsilon_greedy(Q, state, epsilon)            
           
            # take one step corresponding to state, action
            next_state, reward, goal, _ = env.step(action) 
            
            # update reward stats 
            x_reward[ep] += reward                        
                       
            # TD & Q Update
            next_state_q = 12*next_state[0] + next_state[1]
            state_q = 12*state[0] + state[1]            
            best_next_action = np.argmax(Q[next_state_q], axis = 0)     
            td_update = reward + gamma * Q[next_state_q][best_next_action] - Q[state_q][action]                     
            Q[state_q][action] += alpha * td_update
           
            # if goal is reached    
            if goal:                
                x_time.append(t+1)
                break

            # update state               
            state = next_state
            
    return Q, x_time, x_reward
            
                   
def epsilon_greedy(Q, state, eps):
    i = random.uniform(0,1)
    # exploration
    if i <= eps:
        action = random.randint(0,3)                
    # exploitation  
    else:
        state_q = 12*state[0] + state[1]
        action = np.argmax(Q[state_q], axis = 0)
    return action

def plot_policy(env, Q, g):
        plt.rcParams['figure.figsize'] = [8,8]               
        pol = np.argmax(Q, axis = 1)
        shape = (env.height, env.width)
        pol = pol.reshape(shape)
        fig, at = plt.subplots() 
        at.matshow(pol)
        for i in range(env.height):
            for j in range(env.width):
                if env.goal == (j,i):
                    at.text(i,j,''+ g, va='center', ha='center')
                else:                    
                    label = {0:'↑', 1:'➜', 2:'←', 3:'↓' }
                    at.text(i, j, label[int(pol[j,i])], va='center', ha='center')


env = gym.make('gym_foo:foo-v0')
runs = 50
episodes = 500

for g in ['A', 'B', 'C']:

    #g = 'C'         # Change goals here

    if g == 'B':
        env.goal = (2,9)
    elif g == 'C':
        env.goal = (6,7)
    else:
        env.goal = (0,11)

    # Variables for plotting
    x_time = np.zeros(episodes)
    x_reward = np.zeros(episodes)

    Q_count = np.zeros([env.height*env.width, 4])

    for kth_run in range(runs):
        print(kth_run)
        Q, x_t, x_r = qlearning(env, episodes, runs)
        print(Q[12*env.goal[0] + env.goal[1]][:])
        Q_max = np.argmax(Q, axis = 1)
        for a in range(env.height*env.width):        
            Q_count[a][Q_max[a]] += 1
        for u in range(episodes):
            x_time[u] += x_t[u]
            x_reward[u] += x_r[u]

    x_time /= runs
    x_reward /= runs
    y_episode = list(range(episodes))

    # Plotting graphs
    plt.figure()
    plt.plot(y_episode, x_time)
    plt.title('Number of steps to reach goal '+ g +': Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.savefig('QL_'+ g +'1.png')
    #plt.show()

    plt.figure()
    plt.plot(y_episode, x_reward)
    plt.title('Average reward per episode for goal '+ g +': Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig('QL_'+ g +'2.png')
    #plt.show()

    # Plotting Policy
    plt.figure()
    plot_policy(env, Q_count, g)
    plt.title('Policy obtained for goal '+ g +': Q-learning')
    plt.savefig('QL_'+ g +'3.png')
    #plt.show()

