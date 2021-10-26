import network
import event
import generate_user
import block_allocation
import priority_queue
import random
import update_data
import statistic

class Simulator:
    LAMBD = 0.9  # Lambda for users generator

    def __init__(self, logger, step_mode_):
        self.step_mode = step_mode_

        self.rb_al_time = 5  # Time between RB allocation
        self.rb_max_nb = 5  # Maximum RB number for each user
        self.rb_number = 50  # Number of all RBs
        self.clock = 0  # Simulation time

        # Dictionary contains all random generators using in simulation
        self.random_container = self.generate_rand_dict()

        # Statistic object
        self.stats = statistic.Stat()

        self.log = logger
        self.sim_network = network.Network(self.log, self.rb_al_time, self.random_container, self.stats)

        self.av_throughput = 0
        self.av_wait_time = 0
        self.av_qos = 0
        self.update_time = 1



    def run(self, time):

        event_queue = priority_queue.PriorityQueue()

        first_user = generate_user.GenerateUser(self.log, event_queue,
                                                int(self.random_container["time"].expovariate(Simulator.LAMBD)),
                                                self.sim_network,
                                                self.rb_max_nb, self.rb_number, self.rb_al_time, self.random_container,
                                                Simulator.LAMBD)

        initial_rb_allocation = block_allocation.BlockAllocation(self.log, event_queue,
                                                                 self.rb_al_time, self.sim_network, self.rb_max_nb,
                                                                 self.rb_number, self.rb_al_time, self.update_time,
                                                                 self.random_container, self.stats)

        first_data_update = update_data.UpdateData(self.log, event_queue,
                                                   self.rb_al_time + self.update_time, self.sim_network,
                                                   self.rb_max_nb, self.rb_number, self.rb_al_time, self.update_time,
                                                   self.random_container)

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

        self.stats.calc_stats(int(time), self.rb_al_time)
        self.stats.display_stats()

    @staticmethod
    def generate_rand_dict(
            time_seed=12345,
            mean_snr_seed=12345,
            snr_seed=12345,
            rb_nb_seed=12345,
            data_seed=12345
    ):
        # Time between user's appearance
        rand_time = random.Random()

        # Mean SNR for single user
        rand_mean_snr = random.Random()

        # SNR for each RB for single user
        rand_snr = random.Random()

        # Amount of RBs witch user expects
        rand_rb_nb = random.Random()

        # Random data amount for each UE
        rand_data = random.Random()

        # Setting seeds
        rand_time.seed(time_seed)
        rand_mean_snr.seed(mean_snr_seed)
        rand_snr.seed(snr_seed)
        rand_rb_nb.seed(rb_nb_seed)
        rand_data.seed(data_seed)

        rand_dict = {
            "time": rand_time,
            "mean_snr": rand_mean_snr,
            "snr": rand_snr,
            "rb_number": rand_rb_nb,
            "data": rand_data
        }

        return rand_dict
