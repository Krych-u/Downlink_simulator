import event
import priority_queue
import logging


class BlockAllocation(event.Event):
    # Chosen algorithm
    ALGORITHM_TYPE = 0

    def __init__(self, logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_, update_time):
        super().__init__(logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_)
        self.update_time = update_time

    def execute(self):

        if self.ALGORITHM_TYPE == 0:
            # Maximum throughput algorithm

            # self.log.info('Updating received data')

            # Updating received data
            self.sim_network.update_received_data(self.update_time)

            # self.sim_network.display_users_list()

            # RB allocation
            self.log.info("RB allocation")

            i = 0
            for rb in self.sim_network.rb_list:
                if rb.rb_user is not None:
                    i += 1
                    continue
                for rb_user in self.sim_network.users_list:
                    if rb_user.throughput[i] > rb.throughput and rb_user.allocated_rb < self.rb_max_nb:
                        self.sim_network.rb_list[self.sim_network.rb_list.index(rb)].set_rb(rb_user, i)
                if rb.rb_user is not None:
                    rb.rb_user.allocated_rb += 1
                i += 1

            self.sim_network.display_throughputs()
            self.sim_network.display_rbs()

            # --------------------------------------------------
            # Adding next rb_allocation event to event queue
            next_rb_allocation = BlockAllocation(self.log, self.event_queue,
                                                 self.rb_al_time + self.time, self.sim_network,
                                                 self.rb_max_nb, self.rb_number, self.rb_al_time, self.update_time)
            self.event_queue.push(next_rb_allocation, next_rb_allocation.time)
        elif self.ALGORITHM_TYPE == 1:
            pass

        return
