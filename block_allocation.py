import event
import priority_queue
import logging
import user
import numpy as np


class BlockAllocation(event.Event):
    # Chosen algorithm
    ALGORITHM_TYPE = 0

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

    @staticmethod
    def shannon_th(snr):
        return int(user.User.RB_BANDWIDTH * np.log2(1 + BlockAllocation.dec2lin(snr)))

    @staticmethod
    def dec2lin(val):
        return pow(10, val/10)



