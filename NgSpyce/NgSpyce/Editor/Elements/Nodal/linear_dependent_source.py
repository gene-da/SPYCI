from NgSpyce.Editor.Elements.Nodal.nodal_element import NodalElement
from NgSpyce.utilities import format_value as fv

class VoltagrControlledCurrentSource(NodalElement):
    def __init__(
        self,
        name: str,
        out_pos,
        out_neg,
        ctrl_pos,
        ctrl_neg,
        value,
        m=None
    ):
        """
        Create a Voltage-Controlled Current Source (Gxxxx) element.

        Args:
            name (str): Element name (e.g., G1)
            out_pos (str|int): Output positive node
            out_neg (str|int): Output negative node
            ctrl_pos (str|int): Control positive node
            ctrl_neg (str|int): Control negative node
            value (float|str): Transconductance value (Siemens)
            m (float|str, optional): Output multiplier
        """
        formatted_value = fv(value)
        super().__init__(name, [out_pos, out_neg, ctrl_pos, ctrl_neg], formatted_value)
        self.m = fv(m) if m is not None else None

    def get_line(self) -> str:
        """Return SPICE netlist line for VCCS"""
        line = ' '.join([self.name] + self.nodes + [self.value])
        if self.m:
            line += f" m={self.m}"
        return line
    
class VoltageControlledVoltageSource(NodalElement):
    def __init__(self, name, out_pos, out_neg, ctrl_pos, ctrl_neg, gain):
        """
        Voltage-Controlled Voltage Source (E).
        Args:
            name: Element name (e.g., E1)
            out_pos/out_neg: Output terminals
            ctrl_pos/ctrl_neg: Control terminals
            gain: Voltage gain (unitless)
        """
        value = fv(gain)
        super().__init__(name, [out_pos, out_neg, ctrl_pos, ctrl_neg], value)

    def get_line(self) -> str:
        return ' '.join([self.name] + self.nodes + [self.value])
    
class CurrentControlledCurrentSource(NodalElement):
    def __init__(self, name, out_pos, out_neg, vname, gain, m=None):
        """
        Current-Controlled Current Source (F).
        Args:
            vname: Name of voltage source through which current is sensed
            gain: Current gain (unitless)
            m: Optional multiplier
        """
        value = fv(gain)
        super().__init__(name, [out_pos, out_neg, vname], value)
        self.m = fv(m) if m is not None else None

    def get_line(self) -> str:
        line = ' '.join([self.name] + self.nodes + [self.value])
        if self.m:
            line += f" m={self.m}"
        return line
    
class CurrentControlledVoltageSource(NodalElement):
    def __init__(self, name, out_pos, out_neg, vname, gain):
        """
        Current-Controlled Voltage Source (H).
        Args:
            gain: Transresistance (Ohms)
        """
        value = fv(gain)
        super().__init__(name, [out_pos, out_neg, vname], value)

    def get_line(self) -> str:
        return ' '.join([self.name] + self.nodes + [self.value])
    
class PolynomialSource(NodalElement):
    def __init__(self, name, source_type, out_pos, out_neg, control_nodes, coeffs):
        """
        Generic POLY dependent source.

        Args:
            source_type: 'E', 'G', 'F', or 'H'
            control_nodes: list of control node pairs or vnames
            coeffs: list of polynomial coefficients
        """
        poly_order = len(control_nodes) // 2 if source_type in ('E', 'G') else len(control_nodes)
        poly_str = f"POLY({poly_order})"
        value_str = ' '.join(str(fv(c)) for c in coeffs)
        nodes = [out_pos, out_neg] + control_nodes
        super().__init__(name, nodes, f"{poly_str} {value_str}")

    def get_line(self):
        return ' '.join([self.name] + self.nodes + [self.value])
    
if __name__ == '__main__':
    g1 = VoltagrControlledCurrentSource("G1", 2, 0, 5, 0, "0.1")
    print(g1.get_line())
    # G1 N002 N000 N005 N000 0.1

    g2 = VoltagrControlledCurrentSource("G2", 1, 0, 3, 2, "0.5m", m=2)
    print(g2.get_line())
    # G2 N001 N000 N003 N002 0.5m m=2
    
    E1 = VoltageControlledVoltageSource("E1", 2, 3, 14, 1, 2.0)
    # → E1 N002 N003 N014 N001 2
    
    F1 = CurrentControlledCurrentSource("F1", 13, 5, "VSENS", 5, m=2)
    # → F1 N013 N005 VSENS 5 m=2
    
    H1 = CurrentControlledVoltageSource("HX", 5, 17, "VZ", "0.5K")
    # → HX N005 N017 VZ 0.5K