import logging
import random
import simulator
import network


def main():
    # Logger settings
    logger = logging.getLogger('Sim_log')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    # ======================================================================
    #  ------- Simulation settings:
    # Step mode
    step_mode = True
    print("Step mode? [yes/no]: ")
    while True:
        response = input()
        if response == "yes":
            step_mode = True
            break
        elif response == "no":
            step_mode = False
            break
        else:
            print("Wrong answer! Type 'yes' or 'no': ")

    # Simulation parameters:
    end_time = -1

    while int(end_time) < 0:
        print("Set simulation end time [ms]: ")
        end_time = input()

    # ======================================================================
    # ----------Simulation:
    # Simulation parameters:
    rb_number = 50
    network_ = network.Network(logger, 5)
    time_ = 200
    simulator_ = simulator.Simulator(logger, step_mode, rb_number, network_)
    simulator_.run(end_time)


if __name__ == "__main__":
    main()
