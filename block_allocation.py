import event
import priority_queue
import logging


class BlockAllocation(event.Event):
    def __init__(self, logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_):
        super().__init__(logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_)

    def execute(self):

        #self.log.info('Updating received data')

        # Updating received data
        self.sim_network.update_received_data()

        self.sim_network.display_users_list()

        # RB allocation
        self.log.info("RB allocation")
        self.sim_network.clear_rb_list()
        i = 0
        for rb in self.sim_network.rb_list:
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
                                             self.rb_max_nb, self.rb_number, self.rb_al_time)
        self.event_queue.push(next_rb_allocation, next_rb_allocation.time)

        return
