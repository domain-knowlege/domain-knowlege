"""
Evolutionary Strategies module for PyTorch models -- modified from https://github.com/alirezamika/evostra
"""
import copy
from multiprocessing.pool import ThreadPool
import pickle
import time

import numpy as np
import torch


class EvolutionModule:

    def __init__(
        self, 
        weights, 
        reward_func,
        population_size=50,
        batch_size=4,
        sigma=0.1,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        threadcount=4,
        render_test=False,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None
    ):
        np.random.seed(int(time.time()))
        self.weights = weights
        self.reward_function = reward_func
        self.POPULATION_SIZE = population_size
        self.batch_size = batch_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.cuda = cuda
        self.decay = decay
        self.sigma_decay = sigma_decay
        # self.pool = ThreadPool(threadcount)
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path


    def jitter_weights(self, weights, population=[], no_jitter=False):
        new_weights = []
        if no_jitter:
            new_weights = weights.clone()
        else:
            jittered = torch.from_numpy(self.SIGMA * population[0]).float()
            if self.cuda:
                jittered = jittered.cuda()
            new_weights = weights + jittered 
        # for i, param in enumerate(weights):
        #     print(param.data, population)
            
        #     if no_jitter:
        #         new_weights.append(param.data)
        #     else:
        #         jittered = torch.from_numpy(self.SIGMA * population[0][i]).float()
        #         # jittered = torch.from_numpy(population[i]).float()

        #         if self.cuda:
        #             jittered = jittered.cuda()
        #         new_weights.append(param.data + jittered)

        # print('population and weights:', population, new_weights)

        return new_weights


    def run(self, iterations, print_step=10):
        for iteration in range(iterations):

            batch_population = []
            param = self.weights[0]
            for b in range(self.batch_size):
                population = []
                for _ in range(self.POPULATION_SIZE):
                    x = []
                    # for param in self.weights:
                    x.append(np.random.randn(*param.data.size()))
                    population.append(x)
                # for param in self.weights:
                # print(param, self.weights)
                x = []
                x.append(np.array([0.0,0.0,0.0]))
                population.append(x)
                batch_population.append(population)
            # print('population:', population)
            # rewards = self.pool.map(
            #     self.reward_function, 
            #     [self.jitter_weights(copy.deepcopy(self.weights), population=pop) for pop in population]
            # )

            new_batch_weights = []
            # print(self.weights)
            # print(len(batch_population))
            for index in range(self.batch_size):
                weight = self.weights[index]
                batch_weight = [self.jitter_weights(copy.deepcopy(weight), population=pop) for pop in batch_population[index]]
                new_batch_weights.extend(batch_weight)
            # print('#new batch weights: ', len(new_batch_weights), len(batch_weight), len(self.weights))

            rewards, _ = self.reward_function(new_batch_weights)
            # print(rewards)
            if True:
            # if np.std(rewards) != 0:
                # print(rewards, population)
                
                for i in range(self.batch_size):
                    param = self.weights[i]
                    rewards_i = rewards[(self.POPULATION_SIZE+1) * i: (self.POPULATION_SIZE+1) * (i + 1)]
                    # print(len(rewards), self.POPULATION_SIZE)
                    max_value, max_index = torch.max(rewards_i, dim=0)
                    # print(batch_population)
                    # print(rewards)
                    # print(max_value, max_index.item(),batch_population[i][max_index.item()])
                
                    perturb = self.SIGMA * batch_population[i][max_index.item()]# torch.from_numpy(np.array(population[max_index.item()])).float()
                    perturb = torch.tensor(perturb[0]) # pylint: disable=not-callable
                    if self.cuda:
                        perturb = perturb.cuda()
                    param.data = param.data + perturb

                # normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                # # print(normalized_rewards)
                # for index, param in enumerate(self.weights):
                #     A = np.array([p[index] for p in population])
                #     rewards_pop = torch.from_numpy(np.dot(A.T, normalized_rewards).T)[0].float()
                #     if self.cuda:
                #         rewards_pop = rewards_pop.cuda()
                #     param.data = param.data + self.LEARNING_RATE / (self.POPULATION_SIZE * self.SIGMA) * rewards_pop

                #     self.LEARNING_RATE *= self.decay
                #     self.SIGMA *= self.sigma_decay

            if (iteration+1) % print_step == 0:


                new_batch_weights = []
                for index, weight in enumerate(self.weights):
                    batch_weight = [self.jitter_weights(copy.deepcopy(weight), no_jitter=True) for pop in batch_population[index]]
                    new_batch_weights.extend(batch_weight)

                test_reward, _ = self.reward_function(new_batch_weights)
                # print(len(test_reward))
                # test_reward = self.reward_function(
                #     [self.jitter_weights(copy.deepcopy(self.weights), no_jitter=True)], render=self.render_test
                # )
                test_reward_0 = test_reward[0:self.POPULATION_SIZE + 1]
                max_value, max_index = torch.max(test_reward_0, dim=0)

                print('iter %d. reward: %f. rotate: %f translate: %f %f' \
                    % (iteration+1, max_value, self.weights[0][0].item()*180, self.weights[0][1].item()*0.1, self.weights[0][2].item()*0.1))

                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))
                
                # if self.reward_goal and self.consecutive_goal_stopping:
                #     if test_reward >= self.reward_goal:
                #         self.consecutive_goal_count += 1
                #     else:
                #         self.consecutive_goal_count = 0

                #     if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                #         return self.weights

        return self.weights