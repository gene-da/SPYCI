from NgSpyce.Editor.Elements.Nodal.resistor import Resistor, BehavioralResistor, SemiconductorResistor
from NgSpyce.Editor.Elements.Nodal.capacitor import Capacitor, BehavioralCapacitor, SemiconductorCapacitor
from NgSpyce.Editor.Elements.Nodal.inductor import Inductor, MutualInductor, BehavioralInductor
from NgSpyce.Editor.Elements.Nodal.current_controlled_switch import CurrentControlledSwitch
from NgSpyce.Editor.Elements.Nodal.voltage_controlled_switch import VoltageControlledSwitch
from NgSpyce.Editor.Elements.Nodal.independent_source import IndependentSource
from NgSpyce.Editor.Elements.Nodal.source_functions import*
from NgSpyce.Editor.Elements.Nodal.linear_dependent_source import VoltageControlledVoltageSource, VoltagrControlledCurrentSource, CurrentControlledCurrentSource, CurrentControlledVoltageSource


class R(Resistor):
    def __init__(self, name, node1, node2, value, ac=None, m=None, scale=None, temp=None, dtemp=None, tc1=None, tc2=None, tce=None, noisy=None):
        super().__init__(name, node1, node2, value, ac, m, scale, temp, dtemp, tc1, tc2, tce, noisy)

class RSC(SemiconductorResistor):
    def __init__(self, name, node1, node2, value=None, mname=None, l=None, w=None, temp=None, dtemp=None, m=None, ac=None, scale=None, noisy=None):
        super().__init__(name, node1, node2, value, mname, l, w, temp, dtemp, m, ac, scale, noisy)
        
class RBH(BehavioralResistor):
    def __init__(self, name, node1, node2, expression, tc1=None, tc2=None, noisy=None):
        super().__init__(name, node1, node2, expression, tc1, tc2, noisy)

class C(Capacitor):
    def __init__(self, name, node1, node2, value=None, mname=None, m=None, scale=None, temp=None, dtemp=None, tc1=None, tc2=None, ic=None):
        super().__init__(name, node1, node2, value, mname, m, scale, temp, dtemp, tc1, tc2, ic)

class CSC(SemiconductorCapacitor):
    def __init__(self, name, node1, node2, value=None, mname=None, l=None, w=None, m=None, scale=None, temp=None, dtemp=None, ic=None):
        super().__init__(name, node1, node2, value, mname, l, w, m, scale, temp, dtemp, ic)

class CBH(BehavioralCapacitor):
    def __init__(self, name, node1, node2, expression, kind='c', tc1=None, tc2=None):
        super().__init__(name, node1, node2, expression, kind, tc1, tc2)

class L(Inductor):
    def __init__(self, name, node1, node2, value=None, mname=None, nt=None, m=None, scale=None, temp=None, dtemp=None, tc1=None, tc2=None, ic=None):
        super().__init__(name, node1, node2, value, mname, nt, m, scale, temp, dtemp, tc1, tc2, ic)

class LBH(BehavioralInductor):
    def __init__(self, name, node1, node2, expression, tc1=None, tc2=None):
        super().__init__(name, node1, node2, expression, tc1, tc2)

class K(MutualInductor):
    def __init__(self, name, inductor1, inductor2, coupling):
        super().__init__(name, inductor1, inductor2, coupling)

class S(VoltageControlledSwitch):
    def __init__(self, name, node1, node2, control_node_pos, control_node_neg, model, state = None):
        super().__init__(name, node1, node2, control_node_pos, control_node_neg, model, state)

class W(CurrentControlledSwitch):
    def __init__(self, name, node1, node2, vsource_name, model, state = None):
        super().__init__(name, node1, node2, vsource_name, model, state)

class V(IndependentSource):
    def __init__(self, name, node1, node2, dc_value=None, waveform=None, ac_mag=None, ac_phase=None, distof1=None, distof2=None):
        super().__init__(name, node1, node2, dc_value, waveform, ac_mag, ac_phase, distof1, distof2)
        
class I(IndependentSource):
    def __init__(self, name, node1, node2, dc_value=None, waveform=None, ac_mag=None, ac_phase=None, distof1=None, distof2=None):
        super().__init__(name, node1, node2, dc_value, waveform, ac_mag, ac_phase, distof1, distof2)
        
class SINE(Sin):
    def __init__(self, v0, va, freq, td=0, theta=0, phase=0):
        super().__init__(v0, va, freq, td, theta, phase)

class PULSE(Pulse):
    def __init__(self, v1, v2, td=0, tr=None, tf=None, pw=None, per=None, np=None):
        super().__init__(v1, v2, td, tr, tf, pw, per, np)

class EXP(Exponential):
    def __init__(self, v1, v2, td1=0, tau1=None, td2=None, tau2=None):
        super().__init__(v1, v2, td1, tau1, td2, tau2)

class PWL(PieceWiseLinear):
    def __init__(self, *points, r=None, td=None):
        super().__init__(*points, r=r, td=td)
    
class SSFM(SingleFrequencyFM):
    def __init__(self, vo, va, fm, mdi, fc, td=0, phasem=0, phasec=0):
        super().__init__(vo, va, fm, mdi, fc, td, phasem, phasec)

class AM(AmplitudeModulationAM):
    def __init__(self, vo, vmo, vma=1, fm=None, fc=None, td=0, phasem=0, phasec=0):
        super().__init__(vo, vmo, vma, fm, fc, td, phasem, phasec)
        
class TRNOISE(TransientNoiseSource):
    def __init__(self, na=0, nt=0, nalpha=0, namp=0, rtsam=0, rtscapt=0, rtsemt=0):
        super().__init__(na, nt, nalpha, namp, rtsam, rtscapt, rtsemt)
        
class TRRANDOM(RandomVoltage):
    def __init__(self, rtype, ts, td=None, param1=None, param2=None):
        super().__init__(rtype, ts, td, param1, param2)
        
class VCCS(VoltagrControlledCurrentSource):
    def __init__(self, name, out_pos, out_neg, ctrl_pos, ctrl_neg, value, m=None):
        super().__init__(name, out_pos, out_neg, ctrl_pos, ctrl_neg, value, m)

class VCVS(VoltageControlledVoltageSource):
    def __init__(self, name, out_pos, out_neg, ctrl_pos, ctrl_neg, gain):
        super().__init__(name, out_pos, out_neg, ctrl_pos, ctrl_neg, gain)
        
class CCCS(CurrentControlledCurrentSource):
    def __init__(self, name, out_pos, out_neg, vname, gain, m=None):
        super().__init__(name, out_pos, out_neg, vname, gain, m)

class CCVS(CurrentControlledVoltageSource):
    def __init__(self, name, out_pos, out_neg, vname, gain):
        super().__init__(name, out_pos, out_neg, vname, gain)
        