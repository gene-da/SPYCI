from lib import sim
from lib.themes.theme import PlotTheme

import plotly.graph_objs as go
import plotly.io as pio

sim = sim.SpiceProject('output', 'test.cir')

sim.run()

sim.read_raw()

signal = sim.get_data('V(TP1)', 'tran')

sim.add_plot('v(TP1)', 'tran', 'TP1', subplot_id=1)
sim.add_plot('v(TP2)', 'tran', 'TP2', subplot_id=2)

sim.plot(title="Transient Response", theme=PlotTheme.tokyonight())

sim.plot_fft(theme=PlotTheme.tokyonight())