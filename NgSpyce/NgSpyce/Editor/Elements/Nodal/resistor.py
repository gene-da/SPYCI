from NgSpyce.Editor.Elements.Nodal.nodal_element import NodalElement
from NgSpyce.utilities import format_value as fv

class Resistor(NodalElement):
    def __init__(
        self,
        name,
        node1,
        node2,
        value,
        ac=None,
        m=None,
        scale=None,
        temp=None,
        dtemp=None,
        tc1=None,
        tc2=None,
        tce=None,
        noisy=None
    ):
        """
        Initialize a 2-terminal Resistor element for SPICE netlists.

        Args:
            name (str): SPICE element name (e.g., 'R1').
            node1 (Union[str, int]): First terminal node.
            node2 (Union[str, int]): Second terminal node.
            value (Union[int, float, str]): Resistance value in ohms.
            ...
        """

        if node1 is None or node2 is None:
            raise ValueError("Resistor requires exactly two nodes: node1 and node2.")

        # Handle 0-ohm override
        if isinstance(value, (int, float)) and value == 0:
            value = 1e-12
        elif isinstance(value, str) and value.strip() in ["0", "0.0"]:
            value = "1p"

        formatted_value = fv(value)

        # Force exactly two nodes for Resistor
        super().__init__(name, [node1, node2], formatted_value)

        self.ac = fv(ac) if ac is not None else None
        self.m = fv(m) if m is not None else None
        self.scale = fv(scale) if scale is not None else None
        self.temp = fv(temp) if temp is not None else None
        self.dtemp = fv(dtemp) if dtemp is not None else None
        self.tc1 = fv(tc1) if tc1 is not None else None
        self.tc2 = fv(tc2) if tc2 is not None else None
        self.tce = fv(tce) if tce is not None else None
        self.noisy = str(int(bool(noisy))) if noisy is not None else None

    def get_line(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        params = []

        if self.ac:    params.append(f"ac={self.ac}")
        if self.m:     params.append(f"m={self.m}")
        if self.scale: params.append(f"scale={self.scale}")
        if self.temp:  params.append(f"temp={self.temp}")
        if self.dtemp: params.append(f"dtemp={self.dtemp}")
        if self.tc1:   params.append(f"tc1={self.tc1}")
        if self.tc2:   params.append(f"tc2={self.tc2}")
        if self.tce:   params.append(f"tce={self.tce}")
        if self.noisy: params.append(f"noisy={self.noisy}")

        return ' '.join([self.name] + self.nodes + [self.value] + params)
    
class BehavioralResistor(NodalElement):
    def __init__(
        self,
        name,
        node1,
        node2,
        expression,
        tc1=None,
        tc2=None,
        noisy=None
    ):
        """
        Initialize a behavioral resistor dependent on an expression.

        Args:
            name (str): SPICE element name (e.g., 'R1')
            node1 (Union[int, str]): Positive terminal node
            node2 (Union[int, str]): Negative terminal node
            expression (str): Resistance expression, e.g.:
                              'V(rr) < {Vt} ? {R0} : {2*R0}' or '{5k + 50*TEMPER}'
                              Can be passed with or without 'r =' prefix
            tc1 (Optional[Union[str, float]]): First-order temp coefficient
            tc2 (Optional[Union[str, float]]): Second-order temp coefficient
            noisy (Optional[bool]): Whether the resistor generates noise (default False)
        """

        if node1 is None or node2 is None:
            raise ValueError("BehavioralResistor requires exactly two nodes.")

        # Normalize expression
        expr = str(expression).strip()
        if not expr.startswith('r =') and not expr.startswith("'") and not expr.startswith('"'):
            expr = f"r = '{expr}'"
        elif expr.startswith('r =') and not expr.startswith("r = '"):
            # ensure quoted after r =
            expr = expr.replace('r =', "r = '", 1).rstrip() + "'"

        super().__init__(name, [node1, node2], expr)

        self.tc1 = fv(tc1) if tc1 is not None else None
        self.tc2 = fv(tc2) if tc2 is not None else None
        self.noisy = str(int(bool(noisy))) if noisy is not None else None

    def get_line(self) -> str:
        """
        Generate the ngspice netlist line for a behavioral resistor.

        Returns:
            str: Full SPICE netlist line
        """
        params = []
        if self.tc1: params.append(f"tc1={self.tc1}")
        if self.tc2: params.append(f"tc2={self.tc2}")
        if self.noisy: params.append(f"noisy={self.noisy}")

        return ' '.join([self.name] + self.nodes + [self.value] + params)
    
class SemiconductorResistor(NodalElement):
    def __init__(
        self,
        name,
        node1,
        node2,
        value=None,            # Optional: overrides model-based calc
        mname=None,            # Model name (required if no value)
        l=None,                # Length in meters
        w=None,                # Width in meters (optional if DEFW used)
        temp=None,
        dtemp=None,
        m=None,
        ac=None,
        scale=None,
        noisy=None
    ):
        """
        Semiconductor resistor element, supporting geometric and model-based definitions.

        Args:
            name (str): SPICE element name (e.g., 'RMOD')
            node1 (Union[str, int]): Positive terminal node.
            node2 (Union[str, int]): Negative terminal node.
            value (Union[str, float, int], optional): Resistance in ohms (overrides model).
            mname (str, optional): Model name (required if value is not given).
            l (Union[str, float], optional): Length in meters.
            w (Union[str, float], optional): Width in meters (optional, defaults to model).
            temp (Union[str, float], optional): Instance temperature in °C.
            dtemp (Union[str, float], optional): Delta temperature.
            m (Union[str, float], optional): Instance multiplier.
            ac (Union[str, float], optional): AC resistance.
            scale (Union[str, float], optional): Scaling factor.
            noisy (bool, optional): If False, disables noise generation.
        """
        # Validation
        if value is None:
            if mname is None or l is None:
                raise ValueError("SemiconductorResistor requires either 'value', or both 'mname' and 'l'.")

        # Format the value or model line
        if value is not None:
            value_str = fv(value)
        else:
            value_str = mname

        # Pass 2 nodes to base
        super().__init__(name, [node1, node2], value_str)

        # Store optional attributes
        self.l = fv(l) if l is not None else None
        self.w = fv(w) if w is not None else None
        self.temp = fv(temp) if temp is not None else None
        self.dtemp = fv(dtemp) if dtemp is not None else None
        self.m = fv(m) if m is not None else None
        self.ac = fv(ac) if ac is not None else None
        self.scale = fv(scale) if scale is not None else None
        self.noisy = str(int(bool(noisy))) if noisy is not None else None

    def get_line(self) -> str:
        """
        Generate SPICE netlist line for semiconductor resistor.

        Returns:
            str: SPICE-compatible resistor line with model or geometry.
        """
        params = []

        if self.l:     params.append(f"L={self.l}")
        if self.w:     params.append(f"W={self.w}")
        if self.temp:  params.append(f"temp={self.temp}")
        if self.dtemp: params.append(f"dtemp={self.dtemp}")
        if self.m:     params.append(f"m={self.m}")
        if self.ac:    params.append(f"ac={self.ac}")
        if self.scale: params.append(f"scale={self.scale}")
        if self.noisy: params.append(f"noisy={self.noisy}")

        return ' '.join([self.name] + self.nodes + [self.value] + params)
        
if __name__ == '__main__':
    print(Resistor('R1', 1, 2, 1_000_000).get_line())
    print(Resistor('R1', 1, 2, 1_000_000).get_line())                         # Basic resistor → R1 N001 N002 1Meg
    print(Resistor('R2', 1, 2, 1_000, ac=2_000).get_line())                  # AC override → R2 N001 N002 1K ac=2K
    print(Resistor('R3', 3, 4, 100, m=2).get_line())                         # Multiplier → R3 N003 N004 100 m=2
    print(Resistor('R4', 5, 6, 220, scale=1.5).get_line())                   # Scale factor → R4 N005 N006 220 scale=1.5
    print(Resistor('R5', 7, 8, 330, temp=27).get_line())                     # Instance temperature → R5 N007 N008 330 temp=27
    print(Resistor('R6', 9, 10, 470, dtemp=5).get_line())                    # Temperature delta → R6 N009 N010 470 dtemp=5
    print(Resistor('R7', 11, 12, 560, tc1="2m").get_line())                  # First-order temperature coefficient → R7 N011 N012 560 tc1=2m
    print(Resistor('R8', 13, 14, 680, tc2="1.4u").get_line())                # Second-order temp coeff → R8 N013 N014 680 tc2=1.4u
    print(Resistor('R9', 15, 16, 820, tce="700m").get_line())                # Exponential temp coeff → R9 N015 N016 820 tce=700m
    print(Resistor('R10', 17, 18, 1000, noisy=0).get_line())                 # No noise → R10 N017 N018 1K noisy=0
    print(Resistor('R11', 19, 20, 1_500, tc1="2m", tc2="1.4u", tce="500m", noisy=0).get_line())
    print(BehavioralResistor('R1', 'rr', 0, "V(rr) < {Vt} ? {R0} : {2*R0}", tc1="2e-3", tc2="3.3e-6").get_line())
    print(BehavioralResistor('R2', 'r2', 'rr', "{5k + 50*TEMPER}").get_line())
    print(BehavioralResistor('R3', 'no1', 'no2', "5k * rp1", noisy=True).get_line())
    print(SemiconductorResistor('RLOAD', 2, 10, value='10k').get_line())
    print(SemiconductorResistor('RMOD', 3, 7, mname='RMODEL', l='10u', w='1u').get_line())
    print(SemiconductorResistor('RDEV', 'a', 'b', mname='polyR', l='2u', temp=27, noisy=0).get_line())