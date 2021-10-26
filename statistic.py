import matplotlib.pyplot as plt
import numpy as np


class Stat:
    def __init__(self):
        # Stats collecting during simulation
        self.ue_th_list = []       # List of th every UE throughput
        self.sys_th_list = []      # List of th of system
        self.wait_time = []        # How long users were waiting to receive all data
        self.qos = []              # UE satisfaction of transmission :D

        # Stats to calculate after simulation
        self.av_th = 0
        self.av_wait_time = 0
        self.av_qos = 0
        self.rb_all_time_list = []

    def calc_stats(self, end_time, rb_all_time=5):
        self.av_th = sum(self.ue_th_list[0:len(self.ue_th_list)]) / len(self.ue_th_list)
        self.av_wait_time = sum(self.wait_time[0:len(self.wait_time)]) / len(self.wait_time)

        self.rb_all_time_list = np.linspace(rb_all_time, rb_all_time*len(self.sys_th_list), len(self.sys_th_list))

    def display_stats(self):

        plt.plot(self.rb_all_time_list, self.sys_th_list)
        plt.xlabel('Time')
        plt.ylabel('Throughput')
        plt.title('System throughput')
        plt.axis('tight')
        plt.show()

        plt.hist(self.ue_th_list, density=True,  bins=100)
        plt.ylabel('UE throughput [kbps]')
        plt.xlabel('Number of UE')
        plt.show()

        print(f'Average UE throughput: {self.av_th}')
        print(f'Average wait time: {self.av_wait_time}')






