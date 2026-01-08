from NgSpyce.utilities import format_value as fv

class VoltageControlledSwitch:
    def __init__(
        self,
        name: str,
        node1,
        node2,
        control_node_pos,
        control_node_neg,
        model: str,
        state: str = None  # 'ON' or 'OFF'
    ):
        """
        Voltage-controlled switch.

        Args:
            name (str): SPICE element name (e.g., 'S1')
            node1 (+), node2 (-) (str|int): Switch terminals
            control_node_pos (str|int): Positive control terminal
            control_node_neg (str|int): Negative control terminal
            model (str): Switch model name (e.g., 'switch1')
            state (str, optional): Initial switch state: 'ON' or 'OFF'
        """
        self.name = name
        self.nodes = [str(node1), str(node2)]
        self.control_nodes = [str(control_node_pos), str(control_node_neg)]
        self.model = model
        self.state = state.upper() if state else None
        if self.state not in (None, "ON", "OFF"):
            raise ValueError("Switch state must be 'ON', 'OFF', or None")

    def get_line(self):
        parts = [self.name] + self.nodes + self.control_nodes + [self.model]
        if self.state:
            parts.append(self.state.upper())
        return ' '.join(parts)

if __name__ == '__main__':
    s1 = VoltageControlledSwitch("S1", 10, 0, 1, 0, "switch1", state="off")
    print(s1.get_line())
    s1.state='on'
    print(s1.get_line())