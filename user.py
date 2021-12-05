import numpy as np
import block_allocation
import matplotlib.pyplot as plt

class User:
    RB_BANDWIDTH = 180  # Bandwidth of single RB  [kHz]
    MEAN_RB = 4         # Mean rb number

    def __init__(self, data, user_id, rb_nb, rand_cont, start_time):

        self.received_data = 0        # How many data has user received
        self.start_time = start_time  # Time when user appeared in network
        self.rb_amount = rb_nb        # Total number of rbs in network
        self.random_container = rand_cont

        self.mean_snr = self.random_container['mean_snr'].uniform(-10, 10)
        # List of SNR values for each RB
        self.snr_list = self.generate_snr(self.mean_snr, self.rb_amount, self.random_container['rb_number'])
        self.th_list = np.zeros(rb_nb)
        for i in range(rb_nb):
            self.th_list[i] = block_allocation.BlockAllocation.shannon_th(self.snr_list[i])

        self.allocated_snr_list = []  # SNR list of currently allocated RBs
        self.throughput = 0           # Throughput depends on number of allocated RBs and SNR

        self.data = data              # data number
        self.user_id = user_id
        self.rb_number = self.rb_number_generator(User.MEAN_RB, self.random_container['rb_number'])  # Expected RB number
        self.allocated_rb = 0         # Number of currently allocated RBs to this user
        self.allocated_rb_list = []   # list of allocated rbs

        return

    @staticmethod
    def rb_number_generator(mean_value, rand_gen):

        val = int(rand_gen.gauss(mean_value, 1.2))
        if val <= 0:
            val = 1

        return val

    @staticmethod
    def generate_snr(mean_snr, rb_nb, rand_gen):

        phase = rand_gen.uniform(0, 2 * np.pi)
        x = np.linspace(-np.pi, np.pi, rb_nb)
        snr_list = np.sin(x + phase) + mean_snr

        #xx = np.linspace(0, rb_nb-1, rb_nb)

        #plt.plot(xx, snr_list)
        #plt.xlabel('RB ')
        #plt.ylabel('SNR [dB]')
        #plt.axis('tight')
        #plt.show()

        return snr_list

    def calculate_throughput(self):

        th = 0
        for snr in self.allocated_snr_list:
            th += User.RB_BANDWIDTH * np.log2(1 + pow(10, snr/10))

        self.throughput = th
        return int(th)

    def allocation_array(self):
        array = np.zeros(self.rb_amount)
        for i in self.allocated_rb_list:
            array[i] = 1

        return array

