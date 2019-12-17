import time
import gym
import random
env = gym.make('CartPole-v0')
import math
from collections import deque
import  numpy as np
import sys



num_states=96
max_episode_steps = 500


def discritizer(x, portion):

    if x < portion[0]:return 0
    for i in range(len(portion) - 1):
        if portion[i] <= x < portion[i + 1]:return i + 1
    if x >= portion[-1]: return len(portion)

def statTOstate(stat):

    x, x_dot, theta, theta_dot = stat
    x = discritizer(x, [-1.0, 0.0, +1.0])  # 4 states
    x_dot = discritizer(x_dot, [0.0])  # 2 states
    theta = discritizer(theta, [-2 * math.pi / 72, -1 * math.pi / 72, 0.0, +1 * math.pi / 72, +2 * math.pi / 72])  # 6 states
    theta_dot = discritizer(theta_dot, [0.0])  # 2 states
    state = int(x + 4 * x_dot + 8 * theta + 48 * theta_dot)
    return state



def Train():

    # random.seed(1)
    Qval = [[0] * 2 for i in range(96)]
    gama = 0.9
    alpha = 0.01
    epsilon=1
    LastEpisodes = deque()
    LastScores=0
    episode_count=0

    while epsilon>0:

        episode_count += 1
        state1 = statTOstate(env.reset())
        score = 0
        done = False
        step_count =0

        while not (done) and step_count<max_episode_steps:

            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 1)
            else:
                action = Qval[state1].index(max(Qval[state1]))

            state2, reward, done, ttmm = env.step(action)
            state2 = statTOstate(state2)
            step_count+=step_count

            if (done):
                Qval[state1][action] = (1 - alpha) * Qval[state1][action] + alpha * (reward + 0)
            else:
                Qval[state1][action] = (1 - alpha) * Qval[state1][action] + alpha * (reward + gama * max(Qval[state2]))

            state1 = state2

            # env.render()
            score += int(reward)


        LastEpisodes.append(score)
        LastScores += score
        if LastEpisodes.__len__() > 100: LastScores -= LastEpisodes.popleft()
        epsilon = max(0, epsilon - 0.00001)
        # print(Qval)
        if (episode_count % 1000 == 0): print(
            "Episode = " + str(episode_count) + "   ##     Last Episode Score = " + str(
                score) + "   ##     Epsilon = " + str(
                "%.5f" % epsilon) + "   ##     100 Last Episodes Average = " + str("%.2f" % (LastScores / 100)))


    Policy = [0] * num_states
    for i in range(num_states): Policy[i] = Qval[i].index(max(Qval[i]))
    print("Policy : ")
    print(Policy)
    np.save('q_saved', Policy)


def Play():

    Policy = np.load('q_saved.npy')
    print(Policy)
    episode_count=0
    step_count=0

    while True:
        episode_count += 1
        print('******Episode ',episode_count)
        state = statTOstate(env.reset())

        score = 0
        done = False
        step_count=0
        while not(done) and step_count<max_episode_steps:
            #time.sleep(0.017)
            action = Policy[state]
            state, reward, done,ttmmp = env.step(action)
            state=statTOstate(env.reset())
            step_count += 1
            score += int(reward)
            #env.render()  # render current state of environment

        print('Score:',score)

"""
def AproxTrain():

    # random.seed(1)
    gama = 0.9
    alpha = 0.01
    epsilon=1
    LastEpisodes = deque()
    LastScores=0
    episode_count=0

    ww=1
    ww=1
    ww=1
    ww=1

    print(w1)
    while epsilon>0:

        episode_count += 1
        x,x_dot,theta,theta_dot=env.reset()
        score = 0
        done = False
        step_count =0



        while not (done) and step_count<max_episode_steps:

            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 1)
            else:
                w1*x+w2*x_dot+w3*theta+w4*theta_dot
                action = Qval[state1].index(max(Qval[state1]))

            state2, reward, done, ttmm = env.step(action)
            state2 = statTOstate(state2)
            step_count+=step_count

            if (done):
                Qval[state1][action] = (1 - alpha) * Qval[state1][action] + alpha * (reward + 0)
            else:
                Qval[state1][action] = (1 - alpha) * Qval[state1][action] + alpha * (reward + gama * max(Qval[state2]))

            state1 = state2

            # env.render()
            score += int(reward)


        LastEpisodes.append(score)
        LastScores += score
        if LastEpisodes.__len__() > 100: LastScores -= LastEpisodes.popleft()
        epsilon = max(0, epsilon - 0.00001)
        # print(Qval)
        if (episode_count % 1000 == 0): print(
            "Episode = " + str(episode_count) + "   ##     Last Episode Score = " + str(
                score) + "   ##     Epsilon = " + str(
                "%.5f" % epsilon) + "   ##     100 Last Episodes Average = " + str("%.2f" % (LastScores / 100)))


    Policy = [0] * num_states
    for i in range(num_states): Policy[i] = Qval[i].index(max(Qval[i]))
    print("Policy : ")
    print(Policy)
    np.save('q_saved', Policy)
"""

#yadbegir()
#bekhon()


if __name__ == '__main__':

    if(len(sys.argv)==2 and sys.argv[1]=="train"):
        Train()
    else:
        Play()










