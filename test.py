from lib.project import SpiceProject
from lib.sim import Sim
from lib.themes.theme import PlotTheme
from lib.utils import metric_notation as mn

import numpy as np

project = SpiceProject('output', 'test.cir')

# === Transient Analysis ===
project.sim = Sim.tran('0.1u', '1m', '0')
project.run()
project.read_raw()

project.clear_plots()
project.add_plot('v(TP1)', name='TP1 - Transient')
project.plot(title='TP1 - Time Domain')

# === AC Analysis ===
project = SpiceProject('output', 'test.cir')
project.sim = Sim.ac('dec', 100, '1', '2Meg')
project.run()
project.read_raw()

project.clear_plots()
project.add_plot('v(TP1)', name='TP1 - AC Sweep')
project.plot(title='TP1 - Frequency Domain')