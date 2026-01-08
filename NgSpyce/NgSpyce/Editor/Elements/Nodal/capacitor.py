from NgSpyce.Editor.Elements.Nodal.nodal_element import NodalElement
from NgSpyce.utilities import format_value as fv

class Capacitor(NodalElement):
    def __init__(
        self,
        name,
        node1,
        node2,
        value=None,
        mname=None,
        m=None,
        scale=None,
        temp=None,
        dtemp=None,
        tc1=None,
        tc2=None,
        ic=None,
    ):
        """
        Initialize a 2-terminal Capacitor element for SPICE netlists.

        Args:
            name (str): SPICE element name (e.g., 'C1').
            node1 (Union[str, int]): First terminal node (+).
            node2 (Union[str, int]): Second terminal node (-).
            value (Union[int, float, str], optional): Capacitance value in Farads.
            mname (str, optional): Capacitor model name (if using .model).
            m (Union[str, float], optional): Instance multiplier.
            scale (Union[str, float], optional): Scaling factor.
            temp (Union[str, float], optional): Absolute temperature in Celsius.
            dtemp (Union[str, float], optional): Delta temperature.
            tc1 (Union[str, float], optional): First-order temperature coefficient.
            tc2 (Union[str, float], optional): Second-order temperature coefficient.
            ic (Union[str, float], optional): Initial capacitor voltage (IC=...).
        """

        if node1 is None or node2 is None:
            raise ValueError("Capacitor requires exactly two nodes: node1 and node2.")

        # Handle value or model usage
        if value:
            main_value = fv(value)
        elif mname:
            main_value = mname
        else:
            raise ValueError("Capacitor requires either 'value' or 'mname'.")

        # Normalize two-node format for base class
        super().__init__(name, [node1, node2], main_value)

        self.m = fv(m) if m is not None else None
        self.scale = fv(scale) if scale is not None else None
        self.temp = fv(temp) if temp is not None else None
        self.dtemp = fv(dtemp) if dtemp is not None else None
        self.tc1 = fv(tc1) if tc1 is not None else None
        self.tc2 = fv(tc2) if tc2 is not None else None
        self.ic = str(ic).strip() if ic is not None else None

    def get_line(self) -> str:
        """
        Generate the SPICE netlist line for the capacitor.

        Returns:
            str: SPICE-compatible element line.
        """
        params = []

        if self.m:     params.append(f"m={self.m}")
        if self.scale: params.append(f"scale={self.scale}")
        if self.temp:  params.append(f"temp={self.temp}")
        if self.dtemp: params.append(f"dtemp={self.dtemp}")
        if self.tc1:   params.append(f"tc1={self.tc1}")
        if self.tc2:   params.append(f"tc2={self.tc2}")
        if self.ic:    params.append(f"ic={self.ic}")

        return ' '.join([self.name] + self.nodes + [self.value] + params)
    
class BehavioralCapacitor(NodalElement):
    def __init__(
        self,
        name,
        node1,
        node2,
        expression,
        kind='c',  # 'c' for capacitance, 'q' for charge
        tc1=None,
        tc2=None
    ):
        """
        Behavioral capacitor supporting C or Q expressions.

        Args:
            name (str): SPICE element name (e.g., 'C1').
            node1 (Union[str, int]): First terminal node.
            node2 (Union[str, int]): Second terminal node.
            expression (str): Expression string, e.g.:
                'V(cc) < {Vt} ? {Cl} : {Ch}' or
                '1u*(4*atan(V(a,b)/4)*2+V(a,b))/3'
            kind (str): 'c' for capacitance expression, 'q' for charge expression.
            tc1 (Union[str, float], optional): First-order temp coefficient.
            tc2 (Union[str, float], optional): Second-order temp coefficient.
        """
        if node1 is None or node2 is None:
            raise ValueError("BehavioralCapacitor requires exactly two nodes.")
        if kind.lower() not in ('c', 'q'):
            raise ValueError("kind must be either 'c' (capacitance) or 'q' (charge)")

        kind = kind.lower()
        expr = str(expression).strip()

        # Add 'c =' or 'q =' if not explicitly provided
        if not expr.startswith(("c =", "q =")):
            expr = f"{kind} = '{expr}'"
        elif expr.startswith(f"{kind} =") and not expr.startswith(f"{kind} = '"):
            expr = expr.replace(f"{kind} =", f"{kind} = '", 1).rstrip() + "'"

        super().__init__(name, [node1, node2], expr)

        self.tc1 = fv(tc1) if tc1 is not None else None
        self.tc2 = fv(tc2) if tc2 is not None else None

    def get_line(self) -> str:
        """
        Generate the SPICE netlist line for a behavioral capacitor.

        Returns:
            str: SPICE-compatible capacitor definition line.
        """
        params = []
        if self.tc1: params.append(f"tc1={self.tc1}")
        if self.tc2: params.append(f"tc2={self.tc2}")
        return ' '.join([self.name] + self.nodes + [self.value] + params)

class SemiconductorCapacitor(NodalElement):
    def __init__(
        self,
        name,
        node1,
        node2,
        value=None,
        mname=None,
        l=None,
        w=None,
        m=None,
        scale=None,
        temp=None,
        dtemp=None,
        ic=None,
    ):
        """
        Initialize a semiconductor capacitor with geometric/model support.

        Args:
            name (str): SPICE element name (e.g., 'CMOD').
            node1 (Union[int, str]): First terminal node.
            node2 (Union[int, str]): Second terminal node.
            value (Union[int, float, str], optional): Capacitance in Farads. Overrides model.
            mname (str, optional): Capacitor model name.
            l (Union[str, float], optional): Length in meters (required if no value or model CAP).
            w (Union[str, float], optional): Width in meters.
            m (Union[str, float], optional): Instance multiplier.
            scale (Union[str, float], optional): Scaling factor.
            temp (Union[str, float], optional): Absolute temperature.
            dtemp (Union[str, float], optional): Delta temperature.
            ic (Union[str, float], optional): Initial capacitor voltage (IC=...).
        """
        if node1 is None or node2 is None:
            raise ValueError("SemiconductorCapacitor requires two valid nodes.")

        # Determine what gets placed in the main netlist position
        if value is not None:
            main_value = fv(value)
        elif mname is not None:
            main_value = mname
        else:
            raise ValueError("Must specify either 'value' or 'mname'.")

        super().__init__(name, [node1, node2], main_value)

        self.l = fv(l) if l is not None else None
        self.w = fv(w) if w is not None else None
        self.m = fv(m) if m is not None else None
        self.scale = fv(scale) if scale is not None else None
        self.temp = fv(temp) if temp is not None else None
        self.dtemp = fv(dtemp) if dtemp is not None else None
        self.ic = f"{ic}".strip() if ic is not None else None

    def get_line(self) -> str:
        """
        Generate SPICE netlist line for a semiconductor capacitor.

        Returns:
            str: SPICE-compatible capacitor definition line.
        """
        params = []

        if self.l:     params.append(f"L={self.l}")
        if self.w:     params.append(f"W={self.w}")
        if self.m:     params.append(f"m={self.m}")
        if self.scale: params.append(f"scale={self.scale}")
        if self.temp:  params.append(f"temp={self.temp}")
        if self.dtemp: params.append(f"dtemp={self.dtemp}")
        if self.ic:    params.append(f"ic={self.ic}")

        return ' '.join([self.name] + self.nodes + [self.value] + params)
    
if __name__ == '__main__':
    print(Capacitor("C1", 1, 2, "1u").get_line())
    print(Capacitor("C2", 17, 23, "10u", ic="3V").get_line())
    print(Capacitor("C3", 2, 7, mname="cstd").get_line())
    print(Capacitor("C4", "a", "b", "1n", temp=27, scale=1.2, m=3).get_line())
    print(Capacitor("C5", 3, 4, "2n", tc1="0.001", tc2="1.5u").get_line())
    print(BehavioralCapacitor('C1', 'cc', 0, "V(cc) < {Vt} ? {Cl} : {Ch}", tc1="-1e-3", tc2="1.3e-5").get_line())
    print(BehavioralCapacitor('C2', 'a', 'b', "1u*(4*atan(V(a,b)/4)*2+V(a,b))/3", kind='q').get_line())
    print(SemiconductorCapacitor('CLOAD', 2, 10, value='10p').get_line())
    print(SemiconductorCapacitor('CMOD', 3, 7, mname='CMODEL', l='10u', w='1u').get_line())
    print(SemiconductorCapacitor('C3', 'a', 'b', mname='cmodel').get_line())
    print(SemiconductorCapacitor('C4', 5, 6, mname='CDEF', l='3u', w='1.5u', scale=0.9, temp=27, ic=1.2).get_line())