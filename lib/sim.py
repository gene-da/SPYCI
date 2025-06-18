import os
import shutil
import numpy as np
import inspect
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from dataclasses import dataclass
from lib.spicelib.sim.sim_runner import SimRunner
from lib.spicelib.simulators.ngspice_simulator import NGspiceSimulator
from lib.spicelib.editor.spice_editor import SpiceEditor
from lib.spicelib.raw.raw_read import RawRead
from lib.utils import metric_notation as mn
from lib.themes.theme import PlotTheme

@dataclass
class SpiceData:
    name: any
    data: any

class SpiceProject:        
    def __init__(self, output=None, netlist=None, path_to_ngspice=None):
        caller = inspect.stack()[1].filename
        self.script_dir = os.path.dirname(os.path.abspath(caller))

        if path_to_ngspice is None:
            # fallback path — change this as needed for your system
            path_to_ngspice = "/opt/homebrew/bin/ngspice"  # macOS
            # path_to_ngspice = "C:\\Program Files\\...\\ngspice.exe"  # Windows
        self.simulator = NGspiceSimulator.create_from(path_to_ngspice)
        
        self.netlist = None
        self.output = None
        self.runner = None
        self.reader = None
        self.raw = []
        self._output_files = []
        
        if output:
            self.set_output(output)
            self.__clear_output__()
            self.set_runner()
        if netlist:
            self.set_netlist(netlist=netlist)

        self.plot_data = {}

    def add_plot(self, trace: str, sim_type: str, name: str = None, subplot_id: int = 1):
        """
        Add a trace to a subplot group.

        Args:
            trace (str): name of signal (e.g., 'V(out)')
            sim_type (str): 'tran', 'ac', etc.
            name (str): optional label override
            subplot_id (int): the subplot row to assign this trace to
        """
        data_output = self.get_data(trace, sim_type)
        for i, data in enumerate(data_output):
            label = name or f"{data.name}:{trace}"
            if len(data_output) > 1:
                label += f" #{i+1}"

            if subplot_id not in self.plot_data:
                self.plot_data[subplot_id] = []
            self.plot_data[subplot_id].append(SpiceData(name=label, data=data.data))

        print(f"🟢 Added plot(s) for '{trace}' to subplot {subplot_id}.")


    def remove_plot(self, name: str):
        """
        Remove a specific trace from plot memory.
        """
        if name in self.plot_data:
            del self.plot_data[name]
            print(f"🗑️ Removed plot '{name}' from memory.")
        else:
            print(f"⚠️ Plot '{name}' not found in memory.")

    def clear_plots(self):
        """
        Clear all stored plot data.
        """
        self.plot_data.clear()
        print("🧹 Cleared all plot data.")

    def plot_fft(self, title="FFT Spectrum", theme: PlotTheme = None):
        if not self.plot_data:
            print("⚠️ No data to plot. Use add_plot() first.")
            return

        if theme is None:
            theme = PlotTheme.tokyonight()

        subplot_ids = sorted(self.plot_data.keys())
        fig = make_subplots(
            rows=len(subplot_ids),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.02,
            subplot_titles=["" for _ in subplot_ids]
        )

        color_cycle = theme.colors
        color_index = 0

        for idx, subplot_id in enumerate(subplot_ids):
            for trace in self.plot_data[subplot_id]:
                t, y = trace.data
                dt = t[1] - t[0]  # Assume uniform time base
                N = len(y)

                yf = np.fft.rfft(y)
                xf = np.fft.rfftfreq(N, dt)
                magnitude = np.abs(yf)

                color = color_cycle[color_index % len(color_cycle)]
                fig.add_trace(
                    go.Scatter(
                        x=xf,
                        y=magnitude,
                        mode='lines',
                        name=f"{trace.name} FFT",
                        line=dict(color=color, width=2),
                        hovertemplate=f'%{{x:.3g}} Hz, %{{y:.3g}}<extra>{trace.name}</extra>'
                    ),
                    row=idx + 1, col=1
                )
                color_index += 1

        layout = dict(
            title=dict(text=title, font=dict(color=theme.font)),
            paper_bgcolor=theme.background,
            plot_bgcolor=theme.background,
            font=dict(color=theme.font),
            hovermode='x',
            spikedistance=-1,
            showlegend=True,
            legend=dict(font=dict(color=theme.font)),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(
                title="Frequency (Hz)",
                showspikes=False,
                gridcolor=theme.grid,
                color=theme.font
            )
        )

        for i in range(1, len(subplot_ids) + 1):
            layout[f'yaxis{i}'] = dict(
                title="Magnitude",
                showspikes=False,
                showticklabels=True,
                ticks='outside',
                gridcolor=theme.grid,
                color=theme.font
            )

        fig.update_layout(layout)
        fig.show()
    
    def plot(self, title="Simulation", theme: PlotTheme = None, annotations=None):
        if not self.plot_data:
            print("⚠️ No data to plot. Use add_plot() first.")
            return

        if theme is None:
            theme = PlotTheme.tokyonight()

        subplot_ids = sorted(self.plot_data.keys())
        fig = make_subplots(
            rows=len(subplot_ids),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["" for _ in subplot_ids]
        )

        color_cycle = theme.colors
        color_index = 0

        for idx, subplot_id in enumerate(subplot_ids):
            for trace in self.plot_data[subplot_id]:
                x, y = trace.data
                color = color_cycle[color_index % len(color_cycle)]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name=trace.name,
                        line=dict(color=color, width=2),
                        hovertemplate=f'%{{x:.3g}}, %{{y:.3g}}<extra>{trace.name}</extra>'
                    ),
                    row=idx + 1,
                    col=1
                )
                color_index += 1

        layout = dict(
            title=dict(text=title, font=dict(color=theme.font)),
            paper_bgcolor=theme.background,
            plot_bgcolor=theme.background,
            font=dict(color=theme.font),
            hovermode='x',               # Enables horizontal tooltips only
            spikedistance=-1,            # Disables interactive crosshair
            showlegend=True,
            legend=dict(font=dict(color=theme.font)),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(
                showticklabels=False,
                ticks='',
                showspikes=False,
                gridcolor=theme.grid,
                color=theme.font
            ),
            annotations=annotations or []  # <- Tag support
        )

        for i in range(1, len(subplot_ids) + 1):
            layout[f'yaxis{i}'] = dict(
                showspikes=False,
                showticklabels=True,
                ticks='outside',
                gridcolor=theme.grid,
                color=theme.font
            )

        fig.update_layout(layout)
        fig.show()

    def set_output(self, output_path: str):
        self.output = os.path.join(self.script_dir, output_path)
        os.makedirs(self.output, exist_ok=True)
        
        if os.path.exists(self.output):
            print(f"Directory Created:\t✅\t{output_path}")
        else:
            print(f"Failed to create directory '{self.output}'.")
        
    def set_netlist(self, netlist: str):
        if self.output is None:
            raise ValueError('Did not set output directory')

        full_path = os.path.join(self.script_dir, netlist)
        
        if os.path.isfile(full_path):
            print(f'Netlist Found:\t\t✅\t{netlist}')
            self.netlist = SpiceEditor(full_path)
        else:
            raise FileNotFoundError(f'Netlist file not found at: {full_path}')
            
    def set_runner(self):
        if self.output is None:
            raise ValueError(f'Did not set output directory')
        else:
            self.runner = SimRunner(output_folder=self.output, simulator=self.simulator)
            
    def run(self):
        self.runner.run(self.netlist)
        self.runner.wait_completion()
        self.__check_output_files__()
        
    def __clear_output__(self):
        for file in os.listdir(self.output):
            path = os.path.join(self.output, file)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f'⚠️ Failed to delete {path}. Reason: {e}')
            
    def __check_output_files__(self):        
        for file in os.listdir(self.output):
            if file not in self._output_files:
                print(f' - Created:\t\t✅\t{file}')
                self._output_files.append(file)
            
    def read_raw(self):
        count = 0
        for file in os.listdir(self.output):
            if file.endswith('.raw'):
                count += 1
                path = os.path.join(self.output, file)
                raw = RawRead(path)
                self.raw.append(SpiceData(name=f'Sim{count}', data=raw))
        
        # print(self.raw)
                
    def component(self, component: str, value: str=None):
        if value:
            print(f' - Changed:\t\t❗️\t {component}: {mn.tm(self.netlist[component].value)} ➡️  {mn.tm(mn.fm(value))}')
            self.netlist[component].value = mn.fm(value)
        else:
            return mn.tm(self.netlist[component].value)
        
    def get_data(self, trace: str, sim_type: str):
        data_output = []
        count = 0
        for sim in self.raw:
            count += 1
            read = any
            x = any
            if sim_type == 'tran':
                x = np.real(np.array(sim.data.get_trace('time')))
            if sim_type == 'ac':
                x = np.real(np.array(sim.data.get_trace('frequency')))
            
            y = np.array(sim.data.get_trace(trace))
                
            data_output.append(SpiceData(name=f'Sim {count}', data=(x, y)))
        
        return data_output