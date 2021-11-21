import event
import DQL
import priority_queue
import logging
import user
import numpy as np
import random

class BlockAllocation(event.Event):
    # Chosen algorithm
    ALGORITHM_TYPE = 2
    env = DQL.NetworkEnv()
    agent = DQL.DQNAAgent(env)
    episode = 0


    def __init__(self, logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_, update_time, rand_cont, stats):
        super().__init__(logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_, rand_cont)
        self.update_time = update_time
        self.stats = stats

    def execute(self):

        # Updating received data
        self.sim_network.update_received_data(self.update_time, self.time)
        self.sim_network.clear_rb()

        # RB allocation
        if self.ALGORITHM_TYPE == 0:

            self.log.info("RB allocation using maximum throughput algorithm")
            self.maximum_throughput()

        elif self.ALGORITHM_TYPE == 1:
            self.log.info("RB allocation using round robin algorithm")
            self.round_robin()

        elif self.ALGORITHM_TYPE == 2:
            self.log.info("RB allocation using DQL algorithm")
            self.dql_allocation()

        # Update stats - system throughput
        self.stats.sys_th_list.append(self.sim_network.return_sys_th())

        # Display current network status
        self.sim_network.display_snr_list()
        self.sim_network.display_rbs()


        # --------------------------------------------------
        # Adding next rb_allocation event to event queue
        next_rb_allocation = BlockAllocation(self.log, self.event_queue,
                                             self.rb_al_time + self.time, self.sim_network,
                                             self.rb_max_nb, self.rb_number, self.rb_al_time, self.update_time,
                                             self.random_container, self.stats)
        self.event_queue.push(next_rb_allocation, next_rb_allocation.time)

        return

    def maximum_throughput(self):

        for user in self.sim_network.users_list:
            snr_list = []  # List to save
            highest_snr = -100

            i = 0
            for user_rb in user.snr_list:

                if self.sim_network.rb_list[i].is_free():
                    snr_sum = 0
                    potential_highest_snr = []

                    for j in range(user.rb_number):

                        if j+i > self.sim_network.rb_number-1 or not self.sim_network.rb_list[i+j].is_free():
                            break

                        snr_sum += BlockAllocation.dec2lin(user.snr_list[i+j])
                        potential_highest_snr.append(i+j)

                    if snr_sum > highest_snr:
                        highest_snr = snr_sum
                        snr_list = potential_highest_snr

                i += 1

            for s in snr_list:
                self.sim_network.rb_list[s].rb_user = user
                self.sim_network.rb_list[s].snr = user.snr_list[s]
                user.allocated_snr_list.append(user.snr_list[s])
                self.sim_network.rb_list[s].throughput = self.shannon_th(user.snr_list[s])

            user.throughput = user.calculate_throughput()

    def round_robin(self):
        i = 0
        for user in self.sim_network.users_list:
            if i == len(self.sim_network.rb_list) - 1:
                break

            for rb in range(user.rb_number):
                if i == len(self.sim_network.rb_list) - 1:
                    break

                self.sim_network.rb_list[i].rb_user = user
                self.sim_network.rb_list[i].snr = user.snr_list[i]
                user.allocated_snr_list.append(user.snr_list[i])
                self.sim_network.rb_list[i].throughput = self.shannon_th(user.snr_list[i])

                i += 1

            user.throughput = user.calculate_throughput()

        return

    def dql_allocation(self):
        self.episode += 1
        self.agent.tensorboard.step = self.episode
        episode_reward = 0
        step = 1
        self.env.reset(self.sim_network)
        terminal_state = False

        for u in self.sim_network.users_list:
            current_state = self.env.set_observation(u)

            for user_episode in range(u.rb_number):
                if u == self.sim_network.users_list[-1] and user_episode == u.rb_number:
                    terminal_state = True

                if np.random.random() > DQL.epsilon:
                    action = np.argmax(self.agent.get_qs(current_state))
                    print(action)

                else:
                    decision = random.randint(1, 10)

                    # It won't allocate this rb
                    if decision == 1:
                        action = 50
                    # Random rb
                    elif len(u.allocated_rb_list) == 0:
                        action = random.randint(0, self.sim_network.rb_number)
                    else:
                        if u.allocated_rb_list[0] == 0:
                            action = u.allocated_rb_list[-1] + 1
                        elif u.allocated_rb_list[-1] == self.sim_network.rb_number-1:
                            action = u.allocated_rb_list[0] - 1
                        else:
                            decision = random.randint(0, 1)
                            if decision == 0:
                                action = u.allocated_rb_list[-1] + 1
                            else:
                                action = u.allocated_rb_list[0] - 1

                new_state, reward = self.env.step(u, action)
                episode_reward += reward

                if DQL.LEARNING:
                    self.agent.update_replay_memory((current_state, action, reward, new_state))
                    self.agent.train(terminal_state)

                current_state = new_state
                step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        DQL.ep_rewards.append(episode_reward)
        if (not self.episode % DQL.AGGREGATE_STATS_EVERY or self.episode == 1) and DQL.LEARNING:
            average_reward = sum(DQL.ep_rewards[-DQL.AGGREGATE_STATS_EVERY:]) / len(DQL.ep_rewards[-DQL.AGGREGATE_STATS_EVERY:])
            min_reward = min(DQL.ep_rewards[-DQL.AGGREGATE_STATS_EVERY:])
            max_reward = max(DQL.ep_rewards[-DQL.AGGREGATE_STATS_EVERY:])
            self.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=DQL.epsilon)

            # Log
            sys_av_th = self.sim_network.return_sys_th()
            self.stats.sys_av_th_list.append(sys_av_th)

            print(f'[Simulation time: {self.time}] -- Average reward: {int(average_reward)}  --  System throughput: {int(self.stats.return_av_th())}  '
                  f'--  Users: {len(self.sim_network.users_list)}  --  {DQL.epsilon}')

            # Save model, but only when min reward is greater or equal a set value
            self.agent.model.save(
                   f'models/{DQL.MODEL_NAME}__time__{DQL.STARTING_TIME}.model')

        # Decay epsilon
        if DQL.epsilon > DQL.MIN_EPSILON and DQL.LEARNING:
            DQL.epsilon *= DQL.EPSILON_DECAY
            DQL.epsilon = max(DQL.MIN_EPSILON, DQL.epsilon)

    @staticmethod
    def shannon_th(snr):
        return int(user.User.RB_BANDWIDTH * np.log2(1 + BlockAllocation.dec2lin(snr)))

    @staticmethod
    def dec2lin(val):
        return pow(10, val/10)



