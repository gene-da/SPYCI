class Sim:
    @staticmethod
    def tran(tstep="1n", tstop="1u", tstart="0", tmax="1n", uic=False):
        line = f".tran {tstep} {tstop} {tstart} {tmax}"
        if uic:
            line += " uic"
        return ("tran", line)

    @staticmethod
    def ac(type_="dec", npts=100, fstart="1", fstop="1Meg"):
        return ("ac", f".ac {type_} {npts} {fstart} {fstop}")

    @staticmethod
    def dc(source="V1", start="0", stop="5", step="0.1"):
        return ("dc", f".dc {source} {start} {stop} {step}")

    @staticmethod
    def op():
        return ("op", ".op")

    @staticmethod
    def noise(output="v(out)", source="V1", points=100, fstart="1", fstop="1Meg"):
        return ("noise", f".noise {output} {source} {points} {fstart} {fstop}")