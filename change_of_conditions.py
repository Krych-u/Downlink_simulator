import event
import user
import priority_queue
import random


class ChangeOfConditions(event.Event):
    def __init__(self, logger, event_queue_, user_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_):
        super().__init__(logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_)

        self.user = user_

    def execute(self):
        # Checking if user is still on list
        if self.user not in self.sim_network.users_list:
            return

        self.log.info("Change of user's nr: " + str(self.user.user_id) + " propagation conditions")
        for i in range(len(self.user.throughput)):
            self.user.throughput[i] = int(random.uniform(20, 800))

        new_event = ChangeOfConditions(self.log, self.event_queue, self.user, self.time + int(random.expovariate(0.1)),
                                       self.sim_network,
                                       self.rb_max_nb, self.rb_number, self.rb_al_time)
        self.event_queue.push(new_event, new_event.time)

        self.sim_network.display_throughputs()


