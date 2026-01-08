from typing import List

class NodalElement:
    def __init__(self, name: str, nodes: List[str], value: str):
        self.name = name
        self.nodes = [self._format_node(n) for n in nodes]
        self.value = value

    def _format_node(self, node) -> str:
        node_str = str(node).strip()
        if node_str.isdigit():
            return f"N{int(node_str):03d}"
        return node_str

    def get_line(self) -> str:
        return ' '.join([self.name] + self.nodes + [self.value])
    
if __name__ == '__main__':
    from NgSpyce.utilities import format_value as fv
    print(NodalElement('R1', ['N001', 'N002'], fv('47k')).get_line())
    print(NodalElement('R2', ['N002', 'gnd'], fv(10_000)).get_line())
    print(NodalElement('R1', [1, 2, 'gnd'], fv(10_000)).get_line())