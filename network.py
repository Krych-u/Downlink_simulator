import user
import logging


class ResourceBlock:
    def __init__(self):
        self.throughput = 0
        self.rb_user = None
        self.transmission_error = False
        return

    def set_rb(self, user_, i):
        self.rb_user = user_
        self.throughput = user_.throughput[i]


class Network:

    def __init__(self, logger, rb_al_time_):
        self.rb_number = 50
        self.users_list = []
        self.rb_list = []

        for x in range(self.rb_number):
            self.rb_list.append(ResourceBlock())
        self.error_flag = False

        self.log = logger
        self.rb_al_time = rb_al_time_

    def clear_rb_list(self):
        for ite in self.rb_list:
            ite.rb_user = None
            ite.throughput = 0

        for ite in self.users_list:
            ite.allocated_rb_number = 0

    def display_users_list(self):
        self.log.info("Users list: ")
        for user_ite in self.users_list:
            self.log.info("User [ ID: " + str(user_ite.user_id) + " , Data: " + str(user_ite.data) +
                          " , Data received: " + str(user_ite.received_data))

    def update_received_data(self, update_time):
        rb_user = self.rb_list[0].rb_user
        i = 0
        users_to_delete = set()
        while rb_user is not None:
            user_th = self.rb_list[i].throughput

            # It means that user received all data
            if rb_user.data <= (rb_user.received_data + user_th * update_time):
                rb_user.received_data = rb_user.data
                users_to_delete.add(rb_user)

            # And here user did not received all data
            else:
                rb_user.received_data = rb_user.received_data + user_th * update_time

            i += 1
            if i == self.rb_number:
                break
            rb_user = self.rb_list[i].rb_user
        # Clearing rbs
        self.clear_rb(users_to_delete)

        self.display_users_list()

        # Deleting users
        for us in users_to_delete:
            # self.display_users_list()
            # self.log.debug("User id to delete: " + str(us.user_id))
            self.log.info("Deleting user nr " + str(us.user_id))
            self.users_list.remove(us)
        return

    def clear_rb(self, users_to_delete):
        i = 0
        for rb in self.rb_list:
            if rb.rb_user in users_to_delete:
                self.rb_list[i].rb_user = None
                self.rb_list[i].throughput = 0

            i += 1

    def push_user_to_list(self, data, user_id):
        new_user = user.User(data, user_id, self.rb_number)
        self.users_list.append(new_user)
        self.log.info("New user has been generated! [Id: " + str(user_id) + ", Data: " + str(data) + " bit]")
        self.log.info("Users in queue: " + str(len(self.users_list)))

        return

    def display_throughputs(self):
        for us in self.users_list:
            self.log.debug("User ID: " + str(us.user_id) + " ->  " + " ".join(map(str, us.throughput)))

    def display_rbs(self):
        self.log.info("RB list after allocation: ")
        i = 0
        for rb in self.rb_list:
            if rb.rb_user is None:
                break
            self.log.info(" Block nr " + str(i) + " : User: [ ID:" + str(rb.rb_user.user_id) + " , throughput:" + str(rb.throughput) + " ]")
            i += 1







