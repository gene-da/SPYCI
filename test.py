from lib.project import SpiceProject
from lib.sim import Sim
from lib.utils import metric_notation as mn

caps = []
caps_metric = []

for cap in range(60, 70, 10):
    c = cap*mn.fm('100p')
    caps.append(c)
    caps_metric.append(f'{mn.tm(c)}F')

project = SpiceProject('output', 'test.cir')

# === Transient Analysis ===
project.sim = Sim.tran('0.1u', '100u', '50u')
for cap in caps:
    project.component('C1', cap)
    project.run()


project.read_raw()

project.clear_plots()
project.add_plot('V(TP1)', name='TP1 - Transient', labels=caps_metric)
project.add_plot('v(TP1, TP2)', name='R1 Voltage Drop', labels=caps_metric)
project.add_plot('v(TP2)', name='R2 Voltage Drop', labels=caps_metric)

project.plot(title='Time Domain')
