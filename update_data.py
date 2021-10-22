import event


class UpdateData(event.Event):

    def __init__(self, logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_, update_time):
        super().__init__(logger, event_queue_, time_, sim_network_, rb_max_nb_, rb_number_, rb_al_time_)
        self.update_time = update_time

    def execute(self):
        self.log.info(f"Updating data received, time: {self.time}")
        # Update data
        if self.time % self.rb_al_time != 0:
            self.sim_network.update_received_data(self.update_time)


        # Set next update
        next_data_update = UpdateData(self.log, self.event_queue,
                                             self.update_time + self.time, self.sim_network,
                                             self.rb_max_nb, self.rb_number, self.rb_al_time, self.update_time)
        self.event_queue.push(next_data_update, next_data_update.time)
