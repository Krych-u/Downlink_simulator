import event
import random
import priority_queue
import network
import change_of_conditions


class GenerateUser(event.Event):
    user_id = 0

    def __init__(self, logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_, rand_t, lambd):
        super().__init__(logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_)
        self.rand_t = rand_t
        self.lambd = lambd
# 1633029198
    def execute(self):
        self.log.debug("Execute generate user method")
        generate_user_event = GenerateUser(self.log, self.event_queue, self.time + int(self.rand_t.expovariate(self.lambd)),
                                           self.sim_network,
                                           self.rb_max_nb, self.rb_number, self.rb_al_time, self.rand_t, self.lambd)
        self.event_queue.push(generate_user_event, generate_user_event.time)
        self.sim_network.push_user_to_list(int(random.uniform(250, 10000)), self.user_id)

       # new_event = change_of_conditions.ChangeOfConditions(self.log, self.event_queue, self.sim_network.users_list[-1],
       #                                                     self.time + int(random.expovariate(0.1)),
       #                                                     self.sim_network,
       #                                                     self.rb_max_nb, self.rb_number, self.rb_al_time)
       # self.event_queue.push(new_event, new_event.time)

        GenerateUser.user_id += 1
        return
