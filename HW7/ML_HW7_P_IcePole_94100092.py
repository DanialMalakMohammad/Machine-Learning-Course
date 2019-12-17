import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import time

to_freeze_reward = 0
hole_to_end_reward = 0
goal_to_end_reward = 20


def get_state_space_size():
    return environment.observation_space.n


class BiasVarEvaluator:
    def __init__(self, alg, total_iterations, target_observation_ids):
        self.alg = alg
        self.total_iterations = total_iterations
        self.target_observation_ids = target_observation_ids
        self.value_history = []

    def evaluate(self):
        self.value_history.append(np.copy(self.alg.values))
        for iteration in range(self.total_iterations):
            self.alg.train_an_episode(iteration)
            values_copy = np.copy(self.alg.values)
            self.value_history.append(values_copy)
            # todo: compute the mean and var of the window
            #       when there is at least 'window' items in history
            #print(len(self.value_history))


        means_0_0=[]
        variances_0_0=[]
        for iteration in range(19,self.total_iterations):

            mean=np.zeros(64)
            variance=np.zeros(64)

            for jj in range(20):
                mean=mean+self.value_history[iteration-jj]
            mean=mean/20

            for jj in range(20):
                variance=variance+np.power(self.value_history[iteration-jj]-mean,2)
            variance=variance/20

            variances_0_0.append(variance[0])
            means_0_0.append(mean[0])

        plt.xlabel("Iteration")
        plt.ylabel("Value of Start")
        plt.title("Mean of Start ")
        plt.plot(range(19,3000),means_0_0)
        plt.show()

        plt.xlabel("Iteration")
        plt.ylabel("Value of Start")
        plt.title("Variance of Start ")
        plt.plot(range(19, 3000), variances_0_0)
        plt.show()








class PolicyIteration:
    def __init__(self, threshold):
        self.threshold = threshold
        self.values = np.zeros(get_state_space_size())
        # self.current_state=0
        self.pol = np.zeros(64)

        class Policy:
            def __init__(self, values):
                self.values = values

            def choose_action(self, state):
                if np.random.random() > epsilon:
                    P = environment.env.P

                    whatt = np.zeros(4)
                    for i in range(4):
                        for (prob, nextstate, reward, is_terminal) in P[state][i]:
                            whatt[i] += prob * (gamma * self.values[nextstate] + to_freeze_reward)

                    return np.argmax(whatt)
                    # todo: return greedy action

                else:
                    # return environment.action_space.sample()
                    return np.random.randint(0, 4)

        self.policy = Policy

    def train(self):
        P = environment.env.P
        iteration = 0
        diff = np.inf
        while diff > self.threshold:
            last_value = np.copy(self.values)

            for state in range(64):
                if (len(P[state][0]) != 1):
                    whatt = 0
                    for (prob, nextstate, reward, is_terminal) in P[state][self.pol[state]]:
                        whatt += prob * (gamma * last_value[nextstate] + to_freeze_reward)
                        self.values[state] = whatt
                else:
                    if (state == 63):
                        self.values[state] = goal_to_end_reward
                    else:
                        self.values[state] = hole_to_end_reward

            ############################ environment.

            # todo: update state values
            diff = np.sqrt(np.sum(np.square(last_value - self.values)))
            iteration += 1
        print('converged in {} iterations!'.format(iteration))

    def set_pol(self):
        P = environment.env.P
        for state in range(64):
            whatt = np.zeros(4)
            for i in range(4):
                for (prob, nextstate, reward, is_terminal) in P[state][i]:
                    whatt[i] += prob * (gamma * self.values[nextstate] + reward)

            self.pol[state] = np.argmax(whatt)

    def get_policy(self):
        return self.policy(self.values)


# ############################ Estimators ###################################

class ConstantAlphaMCEstimator:
    def __init__(self, policy):
        self.policy = policy
        self.values = None
        self.reset()

    def train_an_episode(self, iteration):
        environment.reset()
        state1 = 0
        states = [0]
        rewards = []
        done = False
        ttmp = None

        while (not (done)):
            state2, reward, done, ttm = environment.step(self.policy.choose_action(state1))

            states.append(state2)
            rewards.append(reward)
            state1 = state2

        last = rewards[-1];
        rewards.pop();
        rewards.append(0);
        rewards.append(last);

        old_values = np.copy(self.values)

        # print("States",states)
        # print("Rewrads",rewards)

        for index in range(len(states)):

            G = 0
            for indd in range(index, len(states)):
                G += rewards[indd] * (gamma ** (indd - index))

            self.values[states[index]] = old_values[states[index]] + alpha * (G - old_values[states[index]])
            # self.values[states[index]]=self.values[states[index]]+alpha*(G-self.values[states[index]])

        # todo: generate a trajectory and update values
        pass

    def reset(self):
        self.values = np.zeros(get_state_space_size()) + bias

    def get_name(self):
        return self.__class__.__name__


class TDZeroEstimator:
    def __init__(self, policy):
        self.policy = policy
        self.values = None
        self.reset()

    def train_an_episode(self, iteration):

        environment.reset()
        state1 = 0
        done = False
        ttmp = None

        while (not (done)):

            state2, reward, done, ttm = environment.step(self.policy.choose_action(state1))
            if (not (done)):
                self.values[state1] += alpha * (reward + gamma * self.values[state2] - self.values[state1])
                state1 = state2
            else:
                self.values[state1] += alpha * (0 + gamma * self.values[state2] - self.values[state1])
                self.values[state2] += alpha * (reward - self.values[state1])
                state1 = state2

        # todo: generate a trajectory and update values
        pass

    def reset(self):
        self.values = np.zeros(get_state_space_size()) + bias

    def get_name(self):
        return self.__class__.__name__


# ############################ End of Estimators ###################################

if __name__ == '__main__':
    environment = gym.make('FrozenLake8x8-v0')
    epsilon = .1
    gamma = .99
    alpha = .15
    bias = 20.0
    window = 40

    # making reward multiplied with 20
    temp_P = environment.env.P
    for s in temp_P:
        for ac in temp_P[s]:
            for index, (pr, next_s, r, d) in enumerate(temp_P[s][ac]):
                temp_P[s][ac][index] = (pr, next_s, r * 20, d)
    environment.env.P = temp_P

    environment.reset()

    ############################### Policy Iteration             ############################

    learner = PolicyIteration(threshold=.01)
    learner.set_pol()
    for jj in range(20):
        learner.train()
        learner.set_pol()

    print()
    print("Policy Achieved by Policy Iteration: ")
    print(learner.pol.reshape(8, 8))
    print()






    ############################### Constant Alpha_MC Estimation  ###########################


    estimator = ConstantAlphaMCEstimator(learner.get_policy())
    evaluator = BiasVarEvaluator(estimator, 3000, [0])
    evaluator.evaluate()

    learner2 = PolicyIteration(threshold=.01)
    learner2.values = estimator.values
    learner2.set_pol()



    print()
    print("Policy Achieved by ConstantAlphaMC: ")
    print(learner2.pol.reshape(8,8))
    print()

    ############################### Constant Alpha_MC Estimation  ###########################



    estimator = TDZeroEstimator(learner.get_policy())
    evaluator = BiasVarEvaluator(estimator, 3000, [0])
    evaluator.evaluate()

    learner3 = PolicyIteration(threshold=.01)
    learner3.values = estimator.values
    learner3.set_pol()


    print()
    print("Policy Achieved by TDZero: ")
    print(learner3.pol.reshape(8, 8))
    print()


    ############################### Play according to Policy  ###########################



    while (True):

        Policy_type = learner ######## learner for Policy Ietration , learner1 for AlphaMC , learner2 for TD

        environment.reset()
        state1 = 0
        done = False
        environment.render()
        while (not (done)):

            state2, reward, done, ttm = environment.step(int(Policy_type.pol[state1]))
            state1 = state2
            environment.render()

            time.sleep(0.25)

        if state2==63: print ("###############","  Reached Goal")
        else : print("##############","  Fell into Hole")
        time.sleep(1)






