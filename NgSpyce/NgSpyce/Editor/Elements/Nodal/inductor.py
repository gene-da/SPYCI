from NgSpyce.Editor.Elements.Nodal.nodal_element import NodalElement
from NgSpyce.utilities import format_value as fv

class Inductor(NodalElement):
    def __init__(
        self,
        name,
        node1,
        node2,
        value=None,
        mname=None,
        nt=None,
        m=None,
        scale=None,
        temp=None,
        dtemp=None,
        tc1=None,
        tc2=None,
        ic=None
    ):
        """
        Initialize a 2-terminal inductor element for SPICE netlists.

        Args:
            name (str): SPICE element name (e.g., 'L1').
            node1 (Union[str, int]): Positive terminal node.
            node2 (Union[str, int]): Negative terminal node.
            value (Union[int, float, str], optional): Inductance in Henry.
            mname (str, optional): Inductor model name.
            nt (Union[int, str], optional): Number of turns (used with model).
            m (Union[int, str], optional): Instance multiplier.
            scale (Union[float, str], optional): Scaling factor.
            temp (Union[float, str], optional): Absolute temperature in °C.
            dtemp (Union[float, str], optional): Delta temperature from nominal.
            tc1 (Union[float, str], optional): First-order temperature coefficient.
            tc2 (Union[float, str], optional): Second-order temperature coefficient.
            ic (Union[float, str], optional): Initial current (IC=...) in Amps.
        """
        if node1 is None or node2 is None:
            raise ValueError("Inductor requires two valid node identifiers.")

        # Primary element line value (either direct or model)
        if value is not None:
            main_value = fv(value)
        elif mname is not None:
            main_value = mname
        else:
            raise ValueError("Must specify either 'value' or 'mname'.")

        super().__init__(name, [node1, node2], main_value)

        self.nt = fv(nt) if nt is not None else None
        self.m = fv(m) if m is not None else None
        self.scale = fv(scale) if scale is not None else None
        self.temp = fv(temp) if temp is not None else None
        self.dtemp = fv(dtemp) if dtemp is not None else None
        self.tc1 = fv(tc1) if tc1 is not None else None
        self.tc2 = fv(tc2) if tc2 is not None else None
        self.ic = f"{ic}".strip() if ic is not None else None

    def get_line(self) -> str:
        """
        Generate the SPICE netlist line for the inductor.

        Returns:
            str: SPICE-compatible inductor definition line.
        """
        params = []

        if self.nt:    params.append(f"nt={self.nt}")
        if self.m:     params.append(f"m={self.m}")
        if self.scale: params.append(f"scale={self.scale}")
        if self.temp:  params.append(f"temp={self.temp}")
        if self.dtemp: params.append(f"dtemp={self.dtemp}")
        if self.tc1:   params.append(f"tc1={self.tc1}")
        if self.tc2:   params.append(f"tc2={self.tc2}")
        if self.ic:    params.append(f"ic={self.ic}")

        return ' '.join([self.name] + self.nodes + [self.value] + params)

class BehavioralInductor(NodalElement):
    def __init__(
        self,
        name: str,
        node1,
        node2,
        expression: str,
        tc1=None,
        tc2=None
    ):
        """
        Behavioral inductor dependent on a dynamic expression.

        Args:
            name (str): SPICE element name (e.g., 'L1').
            node1 (Union[int, str]): Positive terminal node.
            node2 (Union[int, str]): Negative terminal node.
            expression (str): Expression determining inductance.
                Can be voltage/current dependent, e.g.:
                'i(Vm) < {It} ? {Ll} : {Lh}'
            tc1 (Union[str, float], optional): First-order temperature coefficient.
            tc2 (Union[str, float], optional): Second-order temperature coefficient.
        """
        if node1 is None or node2 is None:
            raise ValueError("BehavioralInductor requires exactly two nodes.")

        expr = str(expression).strip()

        if not expr.startswith("L =") and not expr.startswith("'") and not expr.startswith('"'):
            expr = f"L = '{expr}'"
        elif expr.startswith("L =") and not expr.startswith("L = '"):
            expr = expr.replace("L =", "L = '", 1).rstrip() + "'"

        super().__init__(name, [node1, node2], expr)

        self.tc1 = fv(tc1) if tc1 is not None else None
        self.tc2 = fv(tc2) if tc2 is not None else None

    def get_line(self) -> str:
        """
        Generate SPICE netlist line for behavioral inductor.

        Returns:
            str: Netlist line, e.g., "L1 n1 n2 L = 'i(Vm) < {It} ? {Ll} : {Lh}' tc1=..."
        """
        params = []
        if self.tc1: params.append(f"tc1={self.tc1}")
        if self.tc2: params.append(f"tc2={self.tc2}")
        return ' '.join([self.name] + self.nodes + [self.value] + params)
    
class MutualInductor:
    def __init__(self, name: str, inductor1: Inductor, inductor2: Inductor, coupling: float):
        """
        Define pairwise mutual inductance between two inductors.

        Args:
            name (str): SPICE element name (e.g., 'K12', 'KTX')
            inductor1 (Inductor): First inductor instance
            inductor2 (Inductor): Second inductor instance
            coupling (float): Coupling coefficient (0 < K ≤ 1)
        """
        if not (0 < coupling <= 1):
            raise ValueError("Coupling coefficient must be > 0 and ≤ 1")

        self.name = name
        self.ind1_name = inductor1.name
        self.ind2_name = inductor2.name
        self.coupling = fv(coupling)

    def get_line(self) -> str:
        """
        Generate the SPICE netlist line for the coupled inductors.

        Returns:
            str: Netlist line, e.g. 'K12 L1 L2 0.98'
        """
        return f"{self.name} {self.ind1_name} {self.ind2_name} {self.coupling}"
    
if __name__ == '__main__':
    print(Inductor("LLINK", 42, 69, "1u").get_line())
    print(Inductor("LSHUNT", 23, 51, "10u", ic="15.7mA").get_line())
    print(Inductor("L1", 15, 5, mname="indmod1").get_line())
    print(Inductor("Lload", 1, 2, "1u", mname="ind1", nt=10, dtemp=5, tc1="0.001", tc2="0.0001").get_line())
    b1 = BehavioralInductor(
        "L1",
        "l2",
        "lll",
        "i(Vm) < {It} ? {Ll} : {Lh}",
        tc1="-4e-3",
        tc2="6e-5"
    )
    print(b1.get_line())
    l1 = Inductor('L1', 3, 5, 1e-6)
    l2 = Inductor('L1', 2, 4, 200e-6)
    k1 = MutualInductor('K1', l1, l2, 0.999)
    
    print(l1.get_line())
    print(l2.get_line())
    print(k1.get_line())