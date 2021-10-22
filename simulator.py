import network
import event
import generate_user
import block_allocation
import priority_queue
import random
import update_data


class Simulator:
    def __init__(self, logger, step_mode_, rb_nb_, network_):
        self.rb_al_time = 5      # Time between RB allocation
        self.rb_max_nb = 5       # Maximum RB number for each user
        self.rb_number = rb_nb_  # Number of all RBs
        self.clock = 0           # Simulation time

        self.sim_network = network_
        self.step_mode = step_mode_
        self.log = logger

        # Time between user's appearance
        self.rand_time = random.Random()
        self.rand_time.seed(12345)

        self.av_throughput = 0
        self.av_wait_time = 0
        self.av_qos = 0
        self.update_time = 1

    def run(self, time):
        lambd = 0.9
        event_queue = priority_queue.PriorityQueue()

        first_user = generate_user.GenerateUser(self.log, event_queue, int(self.rand_time.expovariate(lambd)), self.sim_network,
                            self.rb_max_nb, self.rb_number, self.rb_al_time, self.rand_time, lambd)

        initial_rb_allocation = block_allocation.BlockAllocation(self.log, event_queue,
                            self.rb_al_time, self.sim_network, self.rb_max_nb, self.rb_number, self.rb_al_time, self.update_time)

        first_data_update = update_data.UpdateData(self.log, event_queue,
                                             self.rb_al_time + self.update_time, self.sim_network,
                                             self.rb_max_nb, self.rb_number, self.rb_al_time, self.update_time)

        # Initialization the simulation model using previously defined event objects
        event_queue.push(first_user, first_user.time)
        event_queue.push(initial_rb_allocation, self.rb_al_time)
        event_queue.push(first_data_update, first_data_update.time)

        # Main loop, actions in every iteration is implemented in execute virtual method
        while int(self.clock) <= int(time):
            event_ = event_queue.pop()
            self.clock = event_.return_time()
            event_.execute()

            self.log.info("Simulation time: " + str(self.clock) + " ms")

            if self.step_mode:
                input("Press any key to continue ... \n")

