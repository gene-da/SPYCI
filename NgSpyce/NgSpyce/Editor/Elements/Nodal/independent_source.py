from NgSpyce.Editor.Elements.Nodal.nodal_element import NodalElement
from NgSpyce.utilities import format_value as fv

class IndependentSource(NodalElement):
    def __init__(
        self,
        name: str,
        node1,
        node2,
        dc_value=None,
        waveform=None,          # e.g., "SIN(0 1 1MEG)", "PWL(0 0 1u 1)", etc.
        ac_mag=None,
        ac_phase=None,
        distof1=None,           # Can be True or (mag, phase)
        distof2=None,           # Same
    ):
        """
        General independent voltage or current source for Ngspice.

        Args:
            name (str): Element name (e.g., 'VIN', 'ISRC')
            node1 (str|int): Positive terminal
            node2 (str|int): Negative terminal
            dc_value (str|float|int, optional): DC / transient flat value
            waveform (str, optional): Transient waveform function (e.g., SIN(...), PWL(...))
            ac_mag (float|str, optional): AC analysis magnitude
            ac_phase (float|str, optional): AC phase (degrees)
            distof1 (bool|tuple, optional): DISTOF1 or (mag, phase)
            distof2 (bool|tuple, optional): DISTOF2 or (mag, phase)
        """
        super().__init__(name, [node1, node2], "")  # value will be composed below
        self.dc_value = fv(dc_value) if dc_value is not None else None
        self.waveform = waveform.strip() if waveform else None
        self.ac_mag = fv(ac_mag) if ac_mag is not None else None
        self.ac_phase = fv(ac_phase) if ac_phase is not None else None
        self.distof1 = distof1
        self.distof2 = distof2

    def _render_distortion(self, tag, value):
        if value is True:
            return tag
        elif isinstance(value, (tuple, list)) and len(value) >= 1:
            mag = fv(value[0])
            phase = fv(value[1]) if len(value) > 1 else "0"
            return f"{tag} {mag} {phase}"
        return None

    def get_line(self):
        parts = [self.name] + self.nodes

        # DC / TRAN / waveform
        if self.waveform:
            if self.dc_value:  # DC prefix only if value is given
                parts += ['DC', self.dc_value, self.waveform]
            else:
                parts.append(self.waveform)
        elif self.dc_value is not None:
            parts += ['DC', self.dc_value]

        # AC
        if self.ac_mag:
            ac_part = ["AC", self.ac_mag]
            if self.ac_phase:
                ac_part.append(self.ac_phase)
            parts += ac_part

        # DISTO
        d1 = self._render_distortion("DISTOF1", self.distof1)
        d2 = self._render_distortion("DISTOF2", self.distof2)
        if d1: parts.append(d1)
        if d2: parts.append(d2)

        return ' '.join(str(p) for p in parts)
    
class VoltageSource(IndependentSource):
    def __init__(
        self,
        name: str,
        node1,
        node2,
        dc_value=None,
        waveform=None,
        ac_mag=None,
        ac_phase=None,
        distof1=None,
        distof2=None,
    ):
        if not name.upper().startswith("V"):
            raise ValueError("VoltageSource name must begin with 'V'")
        super().__init__(name, node1, node2, dc_value, waveform, ac_mag, ac_phase, distof1, distof2)
        
class CurrentSource(IndependentSource):
    def __init__(
        self,
        name: str,
        node1,
        node2,
        dc_value=None,
        waveform=None,
        ac_mag=None,
        ac_phase=None,
        distof1=None,
        distof2=None,
    ):
        if not name.upper().startswith("I"):
            raise ValueError("CurrentSource name must begin with 'I'")
        super().__init__(name, node1, node2, dc_value, waveform, ac_mag, ac_phase, distof1, distof2)
    
if __name__ == '__main__':
    vcc = IndependentSource("VCC", 10, 0, dc_value=6)
    print(vcc.get_line())
    vin = IndependentSource("VIN", 13, 2, dc_value=0.001, ac_mag=1, waveform="SIN(0 1 1MEG)")
    print(vin.get_line())
    isrc = IndependentSource("ISRC", 23, 21, ac_mag=0.333, ac_phase=45.0, waveform="SFFM(0 1 10K 5 1K)")
    print(isrc.get_line())
    vmod = IndependentSource("VMOD", 2, 0, distof2=(0.01,))
    print(vmod.get_line())
    
    v1 = VoltageSource("VIN", 1, 0, dc_value="5", waveform="SIN(0 1 1k)", ac_mag="1")
    print(v1.get_line())

    i1 = CurrentSource("IIN", "n1", "n2", ac_mag="1", distof2=(0.001,))
    print(i1.get_line())
