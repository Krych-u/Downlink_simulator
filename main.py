import logging
import simulator
import network


def main():
    # Logger settings
    logger = logging.getLogger('Sim_log')
    logger.setLevel(logging.FATAL)

    ch = logging.StreamHandler()
    ch.setLevel(logging.FATAL)

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

        try:
            end_time = int(end_time)
        except ValueError:
            print("Wrong input value type! Please try again typing a number: ")
            end_time = -1


    # ======================================================================
    # ----------Simulation:
    # Simulation parameters:
    simulator_ = simulator.Simulator(logger, step_mode)
    simulator_.run(end_time)


if __name__ == "__main__":
    main()