from common_types import *
import dense
import training


def main_loop():
    is_running = True
    while is_running:
        split= input().split(' ')
        cmd = split[0]
        args = split[1:]

        match cmd:
            case 'stop':
                is_running = False

            case 'show':
                for network_name in NETWORKS:
                    print(network_name, NETWORKS[network_name].network_type)

            case 'create':
                if len(args) < 1:
                    print("expects at least 1 argument (name)")
                    continue
                try:
                    hidden = [int(x) for x in args[1:]]
                    new_network = dense.NetworkDense(name=args[0], network_type='dense', hidden=hidden)
                    NETWORKS[new_network.name] = new_network
                except:
                    print("invalid arguments")

            case 'train':
                if len(args) < 2:
                    print("expects at least 2 arguments (name of network, number of epochs)")
                    continue
                training.train(NETWORKS[args[0]], int(args[1]))


if __name__ == "__main__":
    NETWORKS: dict[str, Network] = {}
    main_loop()

