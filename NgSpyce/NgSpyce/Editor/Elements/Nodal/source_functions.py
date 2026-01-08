from NgSpyce.utilities import format_value as fv

class Pulse:
    def __init__(
        self,
        v1,              # Initial value (V or A)
        v2,              # Pulsed value
        td=0,            # Delay time (s)
        tr=None,         # Rise time (s)
        tf=None,         # Fall time (s)
        pw=None,         # Pulse width (s)
        per=None,        # Period (s)
        np=None          # Number of pulses (int), 0 or None = unlimited
    ):
        """
        Construct a PULSE(...) waveform.

        Args:
            v1 (float|str): Initial value (V or A)
            v2 (float|str): Pulsed value (V or A)
            td (float|str): Delay time (default: 0)
            tr (float|str): Rise time (default: TSTEP)
            tf (float|str): Fall time (default: TSTEP)
            pw (float|str): Pulse width (default: TSTOP)
            per (float|str): Period (default: TSTOP)
            np (int|None): Number of pulses (0 or None = infinite)
        """
        self.v1 = fv(v1)
        self.v2 = fv(v2)
        self.td = fv(td)
        self.tr = fv(tr) if tr is not None else None
        self.tf = fv(tf) if tf is not None else None
        self.pw = fv(pw) if pw is not None else None
        self.per = fv(per) if per is not None else None
        self.np = str(np) if isinstance(np, int) and np > 0 else None

    def __str__(self):
        parts = ["PULSE(" + self.v1, self.v2, self.td]

        # Ensure parameters are added in order and grouped
        for val in [self.tr, self.tf, self.pw, self.per, self.np]:
            if val is not None:
                parts.append(val)

        return ' '.join(parts) + ")"
    
class Sin:
    def __init__(
        self,
        v0,             # Offset (V or A)
        va,             # Amplitude (peak, V or A)
        freq,           # Frequency (Hz)
        td=0,           # Delay time (sec)
        theta=0,        # Damping factor (1/sec)
        phase=0         # Phase (radians or degrees, depending on context)
    ):
        """
        Build a SIN(...) waveform for ngspice voltage/current sources.

        Args:
            v0 (float|str): Offset voltage/current (baseline)
            va (float|str): Amplitude
            freq (float|str): Frequency in Hz
            td (float|str): Delay time (default: 0)
            theta (float|str): Damping factor (default: 0)
            phase (float|str): Phase in radians or degrees (default: 0)
        """
        self.v0 = fv(v0)
        self.va = fv(va)
        self.freq = fv(freq)
        self.td = fv(td)
        self.theta = fv(theta)
        self.phase = fv(phase)


    def __str__(self):
        return f"SIN({self.v0} {self.va} {self.freq} {self.td} {self.theta} {self.phase})"
    
class Exponential:
    def __init__(
        self,
        v1,       # Initial value (V or A)
        v2,       # Pulsed value (V or A)
        td1=0,    # Rise delay time (default 0)
        tau1=None,# Rise time constant (default: TSTEP)
        td2=None, # Fall delay time (default: td1 + TSTEP)
        tau2=None # Fall time constant (default: TSTEP)
    ):
        """
        Build an Exponential(...) waveform for ngspice voltage/current sources.

        Args:
            v1 (float|str): Initial voltage/current
            v2 (float|str): Pulsed value
            td1 (float|str): Rise delay time (sec)
            tau1 (float|str): Rise time constant
            td2 (float|str): Fall delay time
            tau2 (float|str): Fall time constant
        """
        self.v1 = fv(v1)
        self.v2 = fv(v2)
        self.td1 = fv(td1)
        self.tau1 = fv(tau1) if tau1 is not None else None
        self.td2 = fv(td2) if td2 is not None else None
        self.tau2 = fv(tau2) if tau2 is not None else None

    def __str__(self):
        parts = [self.v1, self.v2, self.td1]

        # Require remaining values in correct order
        if self.tau1 is not None: parts.append(self.tau1)
        if self.td2 is not None:  parts.append(self.td2)
        if self.tau2 is not None: parts.append(self.tau2)

        return "Exponential(" + ' '.join(parts) + ")"
class PieceWiseLinear:
    def __init__(
        self,
        *points,        # (T1, V1, T2, V2, ...) or list of tuples
        r=None,         # Repeat time (must match a Ti except last)
        td=None         # Time delay
    ):
        """
        Create a PieceWiseLinear(...) waveform definition.

        Args:
            *points: Sequence of (time, value) pairs — flat list or tuples
            r (float|str, optional): Repeat time point (not last)
            td (float|str, optional): Delay before waveform starts
        """
        if len(points) == 1 and isinstance(points[0], list):
            points = points[0]

        # Allow list of tuples
        if all(isinstance(p, tuple) and len(p) == 2 for p in points):
            flat = []
            for t, v in points:
                flat.extend([fv(t), fv(v)])
            self.points = flat
        else:
            if len(points) % 2 != 0:
                raise ValueError("PieceWiseLinear points must be in (time, value) pairs.")
            self.points = [fv(p) for p in points]

        self.r = fv(r) if r is not None else None
        self.td = fv(td) if td is not None else None

    def __str__(self):
        base = "PieceWiseLinear(" + ' '.join(self.points) + ")"
        extras = []
        if self.r is not None:
            extras.append(f"r={self.r}")
        if self.td is not None:
            extras.append(f"td={self.td}")
        return base + (' ' + ' '.join(extras) if extras else "")
    
class SingleFrequencyFM:
    def __init__(
        self,
        vo,         # Offset (V or A)
        va,         # Amplitude
        fm,         # Modulating frequency (Hz)
        mdi,        # Modulation index (unitless)
        fc,         # Carrier frequency (Hz)
        td=0,       # Delay time (sec)
        phasem=0,   # Modulation phase (deg)
        phasec=0    # Carrier phase (deg)
    ):
        """
        Construct a SFFM(...) waveform for ngspice voltage/current sources.

        Args:
            vo (float|str): Offset value
            va (float|str): Amplitude
            fm (float|str): Modulating frequency
            mdi (float|str): Modulation index
            fc (float|str): Carrier frequency
            td (float|str): Delay time
            phasem (float|str): Phase of modulation signal
            phasec (float|str): Phase of carrier
        """
        self.vo = fv(vo)
        self.va = fv(va)
        self.fm = fv(fm)
        self.mdi = fv(mdi)
        self.fc = fv(fc)
        self.td = fv(td)
        self.phasem = fv(phasem)
        self.phasec = fv(phasec)

    def __str__(self):
        return f"SFFM({self.vo} {self.va} {self.fm} {self.mdi} {self.fc} {self.td} {self.phasem} {self.phasec})"
    
class AmplitudeModulationAM:
    def __init__(
        self,
        vo,         # Overall offset (V or A)
        vmo,        # Modulation offset (V or A)
        vma=1,      # Modulation amplitude (V or A)
        fm=None,    # Modulation frequency (Hz)
        fc=None,    # Carrier frequency (Hz)
        td=0,       # Delay (s)
        phasem=0,   # Modulation phase (degrees)
        phasec=0    # Carrier phase (degrees)
    ):
        """
        Construct an AM(...) waveform for ngspice sources.

        Args:
            vo (float|str): Overall signal offset
            vmo (float|str): Modulation signal offset
            vma (float|str): Modulation signal amplitude (default 1)
            fm (float|str): Modulation frequency
            fc (float|str): Carrier frequency
            td (float|str): Delay time
            phasem (float|str): Phase of modulation signal
            phasec (float|str): Phase of carrier
        """
        self.vo = fv(vo)
        self.vmo = fv(vmo)
        self.vma = fv(vma)
        self.fm = fv(fm) if fm is not None else "5/TSTOP"  # default: 5 / TSTOP
        self.fc = fv(fc) if fc is not None else "500/TSTOP"  # default: 500 / TSTOP
        self.td = fv(td)
        self.phasem = fv(phasem)
        self.phasec = fv(phasec)

    def __str__(self):
        return f"AM({self.vo} {self.vmo} {self.vma} {self.fm} {self.fc} {self.td} {self.phasem} {self.phasec})"
    
class TransientNoiseSource:
    def __init__(
        self,
        na=0,         # White noise RMS amplitude
        nt=0,         # White noise timestep
        nalpha=0,     # 1/f Exponentialonent
        namp=0,       # 1/f RMS amplitude
        rtsam=0,      # RTS amplitude
        rtscapt=0,    # RTS capture time
        rtsemt=0      # RTS emission time
    ):
        """
        Create a TRNOISE(...) waveform Exponentialression for voltage or current sources.

        Args:
            na (float|str): White noise amplitude
            nt (float|str): Time step for sampling
            nalpha (float|str): 1/f Exponentialonent
            namp (float|str): 1/f noise RMS amplitude
            rtsam (float|str): RTS amplitude
            rtscapt (float|str): RTS trap capture time
            rtsemt (float|str): RTS trap emission time
        """
        self.na = fv(na)
        self.nt = fv(nt)
        self.nalpha = fv(nalpha)
        self.namp = fv(namp)
        self.rtsam = fv(rtsam)
        self.rtscapt = fv(rtscapt)
        self.rtsemt = fv(rtsemt)

    def __str__(self):
        return f"TRNOISE({self.na} {self.nt} {self.nalpha} {self.namp} {self.rtsam} {self.rtscapt} {self.rtsemt})"
    
class RandomVoltage:
    def __init__(
        self,
        rtype: int,        # 1 = uniform, 2 = gaussian, 3 = Exponentialonential, 4 = Poisson
        ts,                # Duration of each random value
        td=None,           # Optional delay before start
        param1=None,       # Depends on rtype
        param2=None        # Depends on rtype
    ):
        """
        Construct RandomVoltage(...) waveform string for random voltage/current sources.

        Args:
            rtype (int): Random type (1=uniform, 2=gaussian, 3=Exponentialonential, 4=poisson)
            ts (float|str): Sampling duration per value
            td (float|str, optional): Delay before randomization starts
            param1 (float|str, optional): Distribution-specific parameter
            param2 (float|str, optional): Offset or second parameter
        """
        if rtype not in (1, 2, 3, 4):
            raise ValueError("RandomVoltage type must be 1 (uniform), 2 (gaussian), 3 (Exponential), or 4 (poisson)")

        self.rtype = str(rtype)
        self.ts = fv(ts)
        self.td = fv(td) if td is not None else None
        self.param1 = fv(param1) if param1 is not None else None
        self.param2 = fv(param2) if param2 is not None else None

    def __str__(self):
        args = [self.rtype, self.ts]
        if self.td is not None:
            args.append(self.td)
        if self.param1 is not None:
            args.append(self.param1)
        if self.param2 is not None:
            args.append(self.param2)
        return f"RandomVoltage({', '.join(args)})"
    
if __name__ == '__main__':
    from NgSpyce.Editor.Elements.Nodal.independent_source import VoltageSource
    # One-shot pulse: -1 to 1V, delay 2ns, 2ns rise/fall, 50ns width, 100ns period, 5 pulses
    p = Pulse(-1, 1, td="2n", tr="2n", tf="2n", pw="50n", per="100n", np=5)
    print(str(p))
    # PULSE(-1 1 2n 2n 2n 50n 100n 5)

    # Repeating pulse with default rise/fall times
    p2 = Pulse('0', 5, td="1u", pw="10u", per="20u")
    print(str(p2))
    # PULSE(0 5 1u 10u 20u)

    # Use in a VoltageSource
    vsrc = VoltageSource("VIN", 3, 0, waveform=str(p))
    print(vsrc.get_line())
    # VIN N003 N000 PULSE(-1 1 2n 2n 2n 50n 100n 5)
    
    # Basic damped sine wave: 0 V offset, 1 V amplitude, 100 MHz, 1 ns delay, theta=1e10
    sin1 = Sin('0', 1, "100Meg", td="1n", theta="1e10")
    print(str(sin1))
    # SIN(0 1 100Meg 1n 1e10 0)

    # Use in a VoltageSource
    vsrc = VoltageSource("VIN", 3, 0, waveform=str(sin1))
    print(vsrc.get_line())
    # VIN N003 N000 SIN(0 1 100Meg 1n 1e10 0)
    
    pulse2 = Pulse("0mil", 5, td="1u", pw="10u", per="20u")
    print(str(pulse2))
    # PULSE(0.000635 5 1u 10u 20u)

    vsrc2 = VoltageSource("VIN", "N003", "N000", waveform=str(pulse2))
    print(vsrc2.get_line())
    
    # Full 6-parameter Exponentialonential waveform
    Exponential1 = Exponential(-4, -1, td1="2n", tau1="30n", td2="60n", tau2="40n")
    print(str(Exponential1))
    # Exponential(-4 -1 2n 30n 60n 40n)

    # Use in a VoltageSource
    vsrc = VoltageSource("VIN", 3, 0, waveform=str(Exponential1))
    print(vsrc.get_line())
    # VIN N003 N000 Exponential(-4 -1 2n 30n 60n 40n)
    
    # Example from the manual
    PieceWiseLinear = PieceWiseLinear([
        (0, -7), ("10n", -7), ("11n", -3), ("17n", -3),
        ("18n", -7), ("50n", -7)
    ], r="10n", td="15n")

    print(str(PieceWiseLinear))
    # PieceWiseLinear(0 -7 10n -7 11n -3 17n -3 18n -7 50n -7) r=10n td=15n

    # Attach to a voltage source
    vsrc = VoltageSource("VCLOCK", 7, 5, waveform=str(PieceWiseLinear))
    print(vsrc.get_line())
    # VCLOCK N007 N005 PieceWiseLinear(0 -7 10n -7 11n -3 17n -3 18n -7 50n -7) r=10n td=15n
    
    sffm = SingleFrequencyFM(0, 2, 20, 45, "1k", "1m", 0, 0)
    print(str(sffm))
    # SFFM(0 2 20 45 1k 1m 0 0)

    vsrc = VoltageSource("V1", 12, 0, waveform=str(sffm))
    print(vsrc.get_line())
    # V1 N012 N000 SFFM(0 2 20 45 1k 1m 0 0)
    
    am1 = AmplitudeModulationAM(0.5, 2, vma=1.8, fm="20K", fc="5MEG", td="1m")
    print(str(am1))
    # AM(0.5 2 1.8 20K 5MEG 1m 0 0)

    # Attach to voltage source
    vsrc = VoltageSource("V1", 12, 0, waveform=str(am1))
    print(vsrc.get_line())
    # V1 N012 N000 AM(0.5 2 1.8 20K 5MEG 1m 0 0)
    
    # White Gaussian noise
    noise1 = TransientNoiseSource("20n", "0.5n")
    print(str(noise1))
    # TRNOISE(20n 0.5n 0 0 0 0 0)

    # 1/f noise only
    noise2 = TransientNoiseSource(na=0, nt=0, nalpha=1.1, namp="12p")
    print(str(noise2))
    # TRNOISE(0 0 1.1 12p 0 0 0)

    # Combined white + 1/f + RTS noise
    noise3 = TransientNoiseSource("1m", "1u", 1.0, "0.1m", "15m", "22u", "50u")
    print(str(noise3))
    # TRNOISE(1m 1u 1 0.1m 15m 22u 50u)

    # Voltage source with noise
    vsrc = VoltageSource("VNOISE", 1, 0, dc_value=0, waveform=str(noise3))
    print(vsrc.get_line())
    # VNOISE N001 N000 DC 0 TRNOISE(1m 1u 1 0.1m 15m 22u 50u)
    
    # Gaussian noise with std=1, mean=0, updated every 10ms
    r1 = RandomVoltage(2, "10m", td=0, param1=1, param2=0)
    print(str(r1))
    # RandomVoltage(2, 10m, 0, 1, 0)

    # Uniform noise ±0.5 around offset 0.5, every 1μs, delayed 0.5μs
    r2 = RandomVoltage(1, "1u", "0.5u", "0.5", "0.5")
    print(str(r2))
    # RandomVoltage(1, 1u, 0.5u, 0.5, 0.5)

    # Voltage source with random waveform
    vsrc = VoltageSource("VR1", "r1", 0, dc_value=0, waveform=str(r1))
    print(vsrc.get_line())
    # VR1 r1 N000 DC 0 RandomVoltage(2, 10m, 0, 1, 0)