class CurrentControlledSwitch:
    def __init__(
        self,
        name: str,
        node1,
        node2,
        vsource_name: str,
        model: str,
        state: str = None  # 'ON' or 'OFF'
    ):
        """
        Current-controlled switch.

        Args:
            name (str): SPICE element name (e.g., 'W1')
            node1 (+), node2 (-) (str|int): Switch terminals
            vsource_name (str): Name of voltage source being monitored
            model (str): Current switch model name (e.g., 'csw1')
            state (str, optional): Initial switch state: 'ON' or 'OFF'
        """
        self.name = name
        self.nodes = [str(node1), str(node2)]
        self.vsource_name = vsource_name
        self.model = model
        self.state = state.upper() if state else None
        if self.state not in (None, "ON", "OFF"):
            raise ValueError("Switch state must be 'ON', 'OFF', or None")

    def get_line(self):
        parts = [self.name] + self.nodes + [self.vsource_name, self.model]
        if self.state:
            parts.append(self.state.upper())
        return ' '.join(parts)

if __name__ == '__main__':
    w1 = CurrentControlledSwitch("W1", 40, 0, "vm3", "wswitch1", state="ON")
    print(w1.get_line())
    w1.state = 'off'
    print(w1.get_line())
