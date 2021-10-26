import network
import priority_queue
import logging


class Event:
    def __init__(self, logger, event_queue_,  time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_, rand_cont):
        self.sim_network = sim_network_
        self.time = time_
        self.rb_max_nb = rb_max_nb_
        self.rb_number = rb_number_
        self.rb_al_time = rb_al_time_

        self.event_queue = priority_queue.PriorityQueue()
        self.event_queue = event_queue_
        self.log = logger

        self.random_container = dict()
        self.random_container = rand_cont

    def execute(self):
        pass

    def return_time(self):
        return self.time




