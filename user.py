import random

class User:

    def __init__(self, data, user_id, rb_number):

        self.received_data = 0    # How many data has user received
        self.resource_blocks = 0  # How many blocks has user allocated
        self.start_time = 0       # Time when transmission to user has started
        self.throughput = []      # List of throughput for each RB
        for x in range( rb_number):
            self.throughput.append(random.randrange(20, 800))

        self.data = data          # data number
        self.user_id = user_id
        self.rb_number = rb_number
        self.allocated_rb = 0  # Number of currently allocated RBs to this user
        return

    def change_throughput(self, i, th): ...

    def rb_number_update(self, nr): ...

    def set_start_time(self, time): ...

    def all_data_received(self) -> bool: ...

