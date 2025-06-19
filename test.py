from lib import sim
from lib.themes.theme import PlotTheme
from lib.utils import metric_notation as mn

import numpy as np

sim_type = 'ac'

sim = sim.SpiceProject('output', 'test.cir')
sim.run()

# caps = []

# for val in range(6300, 6400, 10):
#     c = val*1e-12
#     caps.append(f'{mn.tm(c)}')

# for val in caps:
#     sim.component('C1', val)
#     sim.run()
    
# count = 1
# for val in caps:
#     print(f'Signal {count} C1 = {val}')
#     count += 1

sim.read_raw()

# sim.add_plot('v(IN)', sim_type, 'Input AM Signal', subplot_id=1)
sim.add_plot('v(TP1)', sim_type, 'Input TP1 Signal', subplot_id=2)

sim.plot(log_x=True)