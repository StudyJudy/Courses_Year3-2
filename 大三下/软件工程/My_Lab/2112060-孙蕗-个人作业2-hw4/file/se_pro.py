"""
Circuit Simulator Module

This module provides classes and functions to simulate logic circuits with various types of logic gates.
It supports AND, OR, XOR, NAND, NOR, and NOT gates. The main function processes user input to define
the circuit and runs simulations based on the input values and output requirements.

Classes:
    LogicGate - Base class for logic gates
    ANDGate - AND logic gate
    ORGate - OR logic gate
    XORGate - XOR logic gate
    NANDGate - NAND logic gate
    NORGate - NOR logic gate
    NOTGate - NOT logic gate
    CircuitSimulator - Class for simulating logic circuits

Functions:
    main - Main function to handle input and output
    process_question - Process each question, read inputs, and run the simulation
    add_devices_to_simulator - Add devices to the circuit simulator
    read_all_inputs - Read input values for all runs
    read_all_output_devices - Read output device configurations for all runs
"""

class LogicGate:
    """Base class for logic gates."""

    def __init__(self, func, inputs):
        """Initialize the gate with a function and input ports."""
        self.func = func
        self.inputs = inputs

    def evaluate(self, inputs, outputs):
        """Evaluate the gate's output based on inputs and previous outputs."""
        raise NotImplementedError

    def get_value(self, port, inputs, outputs):
        """Get the value from an input or output port."""
        if port[0] == 'I':
            return inputs[int(port[1:]) - 1]
        if port[0] == 'O':
            return outputs[int(port[1:])]
        raise ValueError(f"Invalid port: {port}")


class ANDGate(LogicGate):
    """AND logic gate."""

    def evaluate(self, inputs, outputs):
        in_values = [self.get_value(port, inputs, outputs) for port in self.inputs]
        return int(all(in_values))


class ORGate(LogicGate):
    """OR logic gate."""

    def evaluate(self, inputs, outputs):
        in_values = [self.get_value(port, inputs, outputs) for port in self.inputs]
        return int(any(in_values))


class XORGate(LogicGate):
    """XOR logic gate."""

    def evaluate(self, inputs, outputs):
        in_values = [self.get_value(port, inputs, outputs) for port in self.inputs]
        return int(sum(in_values) % 2)


class NANDGate(LogicGate):
    """NAND logic gate."""

    def evaluate(self, inputs, outputs):
        in_values = [self.get_value(port, inputs, outputs) for port in self.inputs]
        return int(not all(in_values))


class NORGate(LogicGate):
    """NOR logic gate."""

    def evaluate(self, inputs, outputs):
        in_values = [self.get_value(port, inputs, outputs) for port in self.inputs]
        return int(not any(in_values))


class NOTGate(LogicGate):
    """NOT logic gate."""

    def evaluate(self, inputs, outputs):
        in_values = [self.get_value(port, inputs, outputs) for port in self.inputs]
        return int(not in_values[0])


class CircuitSimulator:
    """Class for simulating logic circuits."""

    def __init__(self):
        """Initialize the circuit simulator with supported gate classes."""
        self.devices = {}
        self.device_classes = {
            'AND': ANDGate,
            'OR': ORGate,
            'XOR': XORGate,
            'NAND': NANDGate,
            'NOR': NORGate,
            'NOT': NOTGate
        }

    def add_device(self, device_id, func, inputs):
        """Add a logic device to the circuit."""
        if func not in self.device_classes:
            raise ValueError(f"Unsupported device function: {func}")
        self.devices[device_id] = self.device_classes[func](func, inputs)

    def detect_cycle(self):
        """Detect if there is a cycle in the circuit."""
        try:
            graph = {device_id: [] for device_id in self.devices}
            for device_id, device in self.devices.items():
                for port in device.inputs:
                    if port[0] == 'O':
                        neighbor = int(port[1:])
                        graph[neighbor].append(device_id)
            visited = {device_id: False for device_id in self.devices}
            stack = {device_id: False for device_id in self.devices}

            for device_id in self.devices:
                if not visited[device_id]:
                    if self._has_cycle(graph, device_id, visited, stack):
                        return True
            return False
        except ValueError as val_err:
            print(f"ValueError in detect_cycle: {val_err}")
            return False
        except KeyError as key_err:
            print(f"KeyError in detect_cycle: {key_err}")
            return False

    def _has_cycle(self, graph, vertex, visited, stack):
        """Helper method to detect a cycle starting from a given vertex."""
        try:
            visited[vertex] = True
            stack[vertex] = True
            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    if self._has_cycle(graph, neighbor, visited, stack):
                        return True
                elif stack[neighbor]:
                    return True
            stack[vertex] = False
            return False
        except ValueError as val_err:
            print(f"ValueError in _has_cycle: {val_err}")
            return False
        except KeyError as key_err:
            print(f"KeyError in _has_cycle: {key_err}")
            return False

    def run(self, all_inputs, all_output_devices):
        """Run the simulation for a set of inputs and output devices."""
        try:
            all_results = []
            for inputs, output_devices in zip(all_inputs, all_output_devices):
                outputs = self._evaluate_circuit(inputs)
                result = [outputs.get(dev, 0) for dev in output_devices[1:]]
                all_results.append(result)
            return all_results
        except ValueError as val_err:
            print(f"ValueError in run: {val_err}")
            return []
        except KeyError as key_err:
            print(f"KeyError in run: {key_err}")
            return []

    def _evaluate_circuit(self, inputs):
        """Evaluate the entire circuit for a given set of inputs."""
        outputs = {}
        for i in range(1, len(self.devices) + 1):
            device = self.devices[i]
            outputs[i] = device.evaluate(inputs, outputs)
        return outputs


def main():
    """Main function to handle input and output."""
    try:
        num_questions = int(input("Please enter the number of questions: "))
        for _ in range(num_questions):
            process_question()
    except ValueError as val_err:
        print(f"ValueError running the program: {val_err}")
    except Exception as exc:
        print(f"Error running the program: {exc}")


def process_question():
    """Process each question, read inputs, and run the simulation."""
    try:
        num_inputs, num_devices = map(int, input("Please enter the number of inputs and devices: ").split())
        if num_inputs <= 0 or num_devices <= 0:
            raise ValueError(
                "The number of input signals must be non-negative and the number of devices must be a positive integer."
            )

        simulator = CircuitSimulator()
        add_devices_to_simulator(simulator, num_devices)

        if simulator.detect_cycle():
            print("LOOP")
            return

        num_runs = int(input("Please enter the number of runs: "))
        if num_runs <= 0:
            raise ValueError("The number of runs must be a positive integer.")

        all_inputs = read_all_inputs(num_runs, num_inputs)
        all_output_devices = read_all_output_devices(num_runs)

        results = simulator.run(all_inputs, all_output_devices)
        for result in results:
            print(*result)
    except ValueError as val_err:
        print(f"Input error: {val_err}")
    except KeyError as key_err:
        print(f"KeyError processing the circuit: {key_err}")
    except Exception as exc:
        print(f"Error processing the circuit: {exc}")


def add_devices_to_simulator(simulator, num_devices):
    """Add devices to the circuit simulator."""
    for i in range(1, num_devices + 1):
        device_info = input(f"Please enter the information for device {i}: ").split()
        func = device_info[0]
        num_inputs = int(device_info[1])
        if num_inputs < 1:
            raise ValueError(f"The number of inputs for device {i} must be at least 1.")
        inputs = device_info[2:]
        if len(inputs) != num_inputs:
            raise ValueError(f"The number of inputs for device {i} does not match.")
        simulator.add_device(i, func, inputs)


def read_all_inputs(num_runs, num_inputs):
    """Read input values for all runs."""
    all_inputs = []
    for i in range(1, num_runs + 1):
        inputs = list(map(int, input(f"Please enter the input values for run {i}: ").split()))
        if len(inputs) != num_inputs:
            raise ValueError(f"The number of input values for run {i} is incorrect.")
        all_inputs.append(inputs)
    return all_inputs


def read_all_output_devices(num_runs):
    """Read output device configurations for all runs."""
    all_output_devices = []
    for i in range(1, num_runs + 1):
        output_devices = list(map(int, input(f"Please enter the device numbers and corresponding signal quantities to output for run {i}: ").split()))
        num_outputs = output_devices[0]
        if len(output_devices[1:]) != num_outputs:
            raise ValueError(f"The number of output devices for run {i} is incorrect.")
        all_output_devices.append(output_devices)
    return all_output_devices


if __name__ == "__main__":
    main()
