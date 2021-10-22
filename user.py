import random
import numpy as np


class User:
    RB_BANDWIDTH = 180  # Bandwidth of single RB  [kHz]

    def __init__(self, data, user_id, rb_number):

        self.received_data = 0    # How many data has user received
        self.start_time = 0       # Time when transmission to user has started

        self.mean_snr = random.uniform(-10, 10)
        self.snr = self.generate_snr(self.mean_snr)  # List of SNR values for each RB
        self.allocated_snr_list = [] # SNR list of currently allocated RBs
        self.throughput = 0  # Throughput depends on number of allocated RBs and SNR

        self.data = data          # data number
        self.user_id = user_id
        self.rb_number = self.rb_number_generator(rb_number)
        self.allocated_rb = 0     # Number of currently allocated RBs to this user

        return

    def rb_number_generator(self, mean_value):

        val = int(random.gauss(mean_value, 1.2))
        if val <= 0:
            val = 1

        return val

    def generate_snr(self, mean_snr):

        phase = random.uniform(0, 2 * np.pi)
        x = np.linspace(-np.pi, np.pi, 50)
        snr_list = np.sin(x + phase) + mean_snr

        return snr_list

    def calculate_throughput(self):

        th = 0
        for snr in self.allocated_snr_list:
            th += User.RB_BANDWIDTH * np.log2(1 + pow(10, snr/10))

        return th
