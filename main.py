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
                if len(args) == 0:
                    for network_name in NETWORKS:
                        print(network_name, NETWORKS[network_name].network_type)
                    continue
                network = get_network(args[0])
                print(network)

            case 'create':
                if len(args) < 1:
                    print("expects at least 1 argument (name)")
                    continue
                try:
                    hidden = [int(x) for x in args[1:]]
                    new_network = dense.NetworkDense(name=args[0], network_type='dense', hidden=hidden)
                    NETWORKS[new_network.name] = new_network
                    print(new_network)
                except:
                    print("invalid arguments")

            case 'train':
                if len(args) < 2:
                    print("expects at least 2 arguments (name of network, number of epochs)")
                    continue
                training.train(NETWORKS[args[0]], int(args[1]))

            case 'save':
                if len(args) < 2:
                    print("expects at least 2 arguments (name of network, save file path)")
                    continue
                network = get_network(args[0])
                if not network:
                    print(f"Network {args[0]} not found")
                    continue
                network.dump(args[1])

            case 'load':
                if len(args) < 3:
                    print("expects at least 3 arguments (savefile path, network type, network name)")
                    continue
                match args[1]:
                    case 'dense':
                        network = dense.NetworkDense(name=args[2], source=args[0], network_type='dense')
                        if not post_network(network):
                            print(f"Network name {network.name} already exists in current memory")
                    case _:
                        print(f"Invalid network type {args[1]}")


def get_network(name: str) -> Network | None:
    if name not in NETWORKS:
        return None
    return NETWORKS[name]


def post_network(network: Network) -> bool:
    if network.name in NETWORKS:
        return False
    NETWORKS[network.name] = network
    return True


if __name__ == "__main__":
    NETWORKS: dict[str, Network] = {}
    main_loop()

