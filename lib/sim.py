import os
import shutil
import numpy as np
import inspect
import platform
import plotly.graph_objs as go
from scipy.signal import get_window
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
    name: str
    data: tuple  # (x, y)

class SpiceProject:
    def __init__(self, output=None, netlist=None, path_to_ngspice=None):
        caller = inspect.stack()[1].filename
        self.script_dir = os.path.dirname(os.path.abspath(caller))

        if path_to_ngspice is None:
            if platform.system() == "Windows":
                path_to_ngspice = r"C:\\Users\\eugene.dann\\Documents\\Programs\\Spice64\\bin\\ngspice.exe"
            elif platform.system() == "Darwin":
                path_to_ngspice = "/opt/homebrew/bin/ngspice"
            else:
                raise EnvironmentError("Unsupported OS or NGSpice path not provided.")

        self.simulator = NGspiceSimulator.create_from(path_to_ngspice)

        self.netlist = None
        self.output = None
        self.runner = None
        self.reader = None
        self.raw = []
        self._output_files = []
        self.plot_data = {}

        if output:
            self.set_output(output)
            self.__clear_output__()
            self.set_runner()
        if netlist:
            self.set_netlist(netlist=netlist)

    def set_output(self, output_path: str):
        self.output = os.path.join(self.script_dir, output_path)
        os.makedirs(self.output, exist_ok=True)
        print(f"Directory Created:\t✅\t{output_path}")

    def set_netlist(self, netlist: str):
        full_path = os.path.join(self.script_dir, netlist)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f'Netlist not found at {full_path}')
        print(f'Netlist Found:\t\t✅\t{netlist}')
        self.netlist = SpiceEditor(full_path)

    def set_runner(self):
        if not self.output:
            raise ValueError("Output directory not set")
        self.runner = SimRunner(output_folder=self.output, simulator=self.simulator)

    def run(self):
        self.runner.run(self.netlist)
        self.runner.wait_completion()
        self.__check_output_files__()

    def __clear_output__(self):
        for file in os.listdir(self.output):
            path = os.path.join(self.output, file)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

    def __check_output_files__(self):
        for file in os.listdir(self.output):
            if file not in self._output_files:
                print(f" - Created:\t\t✅\t{file}")
                self._output_files.append(file)

    def read_raw(self):
        self.raw.clear()
        for file in os.listdir(self.output):
            if file.endswith('.raw'):
                path = os.path.join(self.output, file)
                raw = RawRead(path)
                self.raw.append(SpiceData(name=file, data=raw))

    def component(self, component: str, value: str = None):
        if value:
            print(f' - Changed:\t\t❗️\t {component}: {mn.tm(self.netlist[component].value)} ➡️  {mn.tm(mn.fm(value))}')
            self.netlist[component].value = mn.fm(value)
        else:
            return mn.tm(self.netlist[component].value)

    def __normalize_time(self, t):
        dt = np.mean(np.diff(t))
        total = t[-1] - t[0]
        if total > 1e-6:
            return t
        if 1e-8 < dt < 1e-3:
            return t
        if dt > 1e-3:
            return t * 1e-3
        if 1e-10 < dt <= 1e-8:
            return t * 1e-6
        if dt <= 1e-10:
            return t * 1e-9
        return t

    def get_data(self, trace: str, sim_type: str):
        data_out = []
        for sim in self.raw:
            raw = sim.data

            # Select X-axis
            if sim_type == 'tran':
                x = self.__normalize_time(np.array(raw.get_trace('time')))
            elif sim_type in ['ac', 'noise']:
                x = np.array(raw.get_trace('frequency'))
            elif sim_type == 'dc':
                x = np.array(raw.get_trace('V-sweep'))
            elif sim_type == 'op':
                x = np.array([0])
            else:
                raise ValueError(f"Unsupported sim_type: {sim_type}")

            # Get Y-axis data
            y = np.array(raw.get_trace(trace))

            # Handle complex data
            # Handle complex data
            if np.iscomplexobj(y):
                if not np.all(np.isreal(y)):
                    print(f"ℹ️ Complex values detected in '{trace}', converting to magnitude.")
                y = np.abs(y)

            # Clean up for Plotly
            x = np.real_if_close(x, tol=1000).astype(float)
            y = np.real_if_close(y, tol=1000).astype(float)

            # Debug: confirm types and contents
            # print(f"Trace '{trace}' sample: {y[:5]} | dtype: {y.dtype}")

            data_out.append(SpiceData(name=sim.name, data=(x, y)))
        return data_out



    def add_plot(self, trace: str, sim_type: str, name: str = None, subplot_id: int = 1):
        traces = self.get_data(trace, sim_type)
        if subplot_id not in self.plot_data:
            self.plot_data[subplot_id] = []
        for i, tr in enumerate(traces):
            label = name or f"{tr.name}:{trace}"
            if len(traces) > 1:
                label += f" #{i+1}"
            self.plot_data[subplot_id].append(SpiceData(name=label, data=tr.data))
        print(f"🟢 Added plot(s) for '{trace}' to subplot {subplot_id}.")

    def remove_plot(self, name: str):
        removed = False
        for k in list(self.plot_data.keys()):
            self.plot_data[k] = [p for p in self.plot_data[k] if p.name != name]
            if not self.plot_data[k]:
                del self.plot_data[k]
                removed = True
        if removed:
            print(f"🗑️ Removed plot '{name}' from memory.")
        else:
            print(f"⚠️ Plot '{name}' not found in memory.")

    def clear_plots(self):
        self.plot_data.clear()
        print("🧹 Cleared all plot data.")

    def plot(self, title="Simulation", theme: PlotTheme = None, log_x: bool = False):
        if not self.plot_data:
            print("⚠️ No data to plot. Use add_plot() first.")
            return

        if theme is None:
            theme = PlotTheme.tokyonight()

        subplot_ids = sorted(self.plot_data)
        subplot_count = len(subplot_ids)
        fig_height = min(max(800, subplot_count * 400), 1600)

        fig = make_subplots(rows=subplot_count, cols=1, shared_xaxes=True)
        color_cycle = theme.colors
        color_index = 0

        for i, sp_id in enumerate(subplot_ids):
            for trace in self.plot_data[sp_id]:
                x, y = trace.data
                color = color_cycle[color_index % len(color_cycle)]
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=trace.name,
                    line=dict(color=color, width=2)
                ), row=i + 1, col=1)
                color_index += 1

        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=fig_height,
            margin=dict(t=40, b=20, l=40, r=20),
            paper_bgcolor=theme.background,
            plot_bgcolor=theme.background,
            font=dict(color=theme.font)
        )

        # 🔍 Log X-Axis Handling
        if not log_x:
            should_use_log_x = False
            x_all = []

            for subplot in self.plot_data.values():
                for trace in subplot:
                    x = trace.data[0]
                    if np.all(x > 0):
                        x_all.append(x)

            if x_all:
                x_flat = np.concatenate(x_all)
                if x_flat.max() / x_flat.min() >= 100:
                    should_use_log_x = True
            else:
                should_use_log_x = False
        else:
            should_use_log_x = True

        if should_use_log_x:
            for i in range(len(subplot_ids)):
                fig.update_xaxes(type='log', row=i + 1, col=1)
            print("🔁 X-axis set to log scale.")

        fig.show()

    # def plot_fft(self, title="FFT Spectrum", theme: PlotTheme = None, points_per_decade: int = 100,
    #             freq_range: tuple = None, pad_factor: int = 10, db_scale: bool = True, auto_peak_zoom: bool = False):
    #     if not self.plot_data:
    #         print("⚠️ No data to plot. Use add_plot() first.")
    #         return

    #     if theme is None:
    #         theme = PlotTheme.tokyonight()

    #     subplot_ids = sorted(self.plot_data)
    #     trace_count = sum(len(self.plot_data[sid]) for sid in subplot_ids)
    #     fig_height = max(800, trace_count * 300)

    #     fig = make_subplots(rows=len(subplot_ids), cols=1, shared_xaxes=False)
    #     color_cycle = theme.colors
    #     color_index = 0

    #     for idx, subplot_id in enumerate(subplot_ids):
    #         for trace in self.plot_data[subplot_id]:
    #             t, y = trace.data
    #             N = len(y)
    #             dt = (t[-1] - t[0]) / (N - 1)
    #             fs = 1 / dt
    #             T = N * dt
    #             df = 1 / T

    #             window = np.hanning(N)
    #             y_windowed = y * window

    #             Npad = pad_factor * N
    #             yf = np.fft.rfft(y_windowed, n=Npad)
    #             xf = np.fft.rfftfreq(Npad, dt)

    #             scale = np.sum(window) / N
    #             magnitude = np.abs(yf) / (N / 2) / scale
    #             magnitude[0] /= 2

    #             print(f"📊 Trace: {trace.name}")
    #             print(f" - Sample count (N): {N}")
    #             print(f" - Time step (dt): {dt:.3e} s")
    #             print(f" - Sample rate (fs): {fs:.3e} Hz")
    #             print(f" - Total duration (T): {T:.6f} s")
    #             print(f" - FFT points: {Npad}")
    #             print(f" - FFT resolution (df): {df:.3f} Hz\n")

    #             if db_scale:
    #                 magnitude = 20 * np.log10(magnitude + 1e-12)

    #             # === FREQ RANGE + INTERPOLATION ===
    #             if freq_range:
    #                 fmin, fmax = map(mn.fm, freq_range)
    #                 mask = (xf >= fmin) & (xf <= fmax)
    #                 xf_zoom = xf[mask]
    #                 mag_zoom = magnitude[mask]

    #                 if points_per_decade and len(xf_zoom) > 5:
    #                     from scipy.interpolate import interp1d
    #                     try:
    #                         log_f = np.logspace(np.log10(fmin), np.log10(fmax),
    #                                             num=points_per_decade * int(np.log10(fmax / fmin)))
    #                         interpolator = interp1d(xf_zoom, mag_zoom, bounds_error=False, fill_value=0)
    #                         xf = log_f
    #                         magnitude = interpolator(log_f)
    #                     except Exception as e:
    #                         print(f"⚠️ Interpolation fallback: {e}")
    #                         xf = xf_zoom
    #                         magnitude = mag_zoom
    #                 else:
    #                     xf = xf_zoom
    #                     magnitude = mag_zoom

    #             elif auto_peak_zoom:
    #                 peak_index = np.argmax(magnitude)
    #                 f_peak = xf[peak_index]
    #                 spread = 5 * df
    #                 fmin = f_peak - spread
    #                 fmax = f_peak + spread
    #                 mask = (xf >= fmin) & (xf <= fmax)
    #                 xf = xf[mask]
    #                 magnitude = magnitude[mask]

    #             # === PLOT ===
    #             color = color_cycle[color_index % len(color_cycle)]
    #             fig.add_trace(go.Scatter(
    #                 x=xf,
    #                 y=magnitude,
    #                 mode='lines',
    #                 name=f"{trace.name} FFT",
    #                 line=dict(color=color, width=2)
    #             ), row=idx+1, col=1)

    #             # Peak marker
    #             peak_index = np.argmax(magnitude)
    #             f_peak = xf[peak_index]
    #             fig.add_shape(
    #                 type="line",
    #                 x0=f_peak, x1=f_peak,
    #                 y0=min(magnitude), y1=max(magnitude),
    #                 line=dict(color=color, width=1, dash='dot'),
    #                 row=idx+1, col=1
    #             )
    #             fig.add_annotation(
    #                 text=f"{f_peak:.3g} Hz",
    #                 x=f_peak,
    #                 y=max(magnitude),
    #                 showarrow=False,
    #                 font=dict(color=color),
    #                 xanchor='left',
    #                 row=idx+1, col=1
    #             )

    #             color_index += 1

    #     fig.update_layout(
    #         title=title,
    #         template='plotly_dark',
    #         height=fig_height,
    #         margin=dict(t=40, b=20, l=40, r=20),
    #         paper_bgcolor=theme.background,
    #         plot_bgcolor=theme.background,
    #         font=dict(color=theme.font)
    #     )
    #     fig.show()
    
    def best_fft_window(fs_hz: float, resolution_hz: float) -> int:
        """
        Returns the number of samples needed for a given FFT frequency resolution.
        
        Parameters:
        fs_hz (float): Sampling rate in Hz
        resolution_hz (float): Desired frequency resolution (bin width) in Hz

        Returns:
        int: Minimum number of samples needed for that resolution
        """
        return int(fs_hz / resolution_hz)
    
    def plot_fft(
        self,
        title="FFT Spectrum",
        theme: PlotTheme = None,
        window_type='hann',            # 'hann', 'hamming', 'blackmanharris', 'triang', 'rect', 'raw'
        mag_mode='db',                 # 'db', 'linear', 'power'
        freq_range: tuple = ("20k", "2M"),
        resolution_hz: float = None,   # Optional: desired FFT bin resolution in Hz
        pad_factor: int = 2,
        max_points: int = 2000,
    ):
        def best_fft_window(fs_hz: float, resolution_hz: float) -> int:
            return int(fs_hz / resolution_hz)

        if not self.plot_data:
            print("⚠️ No data to plot. Use add_plot() first.")
            return

        if theme is None:
            theme = PlotTheme.tokyonight()

        subplot_ids = sorted(self.plot_data)
        trace_count = sum(len(self.plot_data[sid]) for sid in subplot_ids)
        fig_height = max(800, trace_count * 300)
        fig = make_subplots(rows=len(subplot_ids), cols=1, shared_xaxes=False)
        color_cycle = theme.colors
        color_index = 0

        fmin, fmax = map(mn.fm, freq_range)
        target_bw = max(fmax - fmin, 10e3)
        min_duration = max(1 / target_bw, 10e-6)

        for idx, subplot_id in enumerate(subplot_ids):
            for trace in self.plot_data[subplot_id]:
                t, y = trace.data
                N = len(y)
                dt = (t[-1] - t[0]) / (N - 1)
                fs = 1 / dt

                if resolution_hz:
                    desired_N = best_fft_window(fs, resolution_hz)
                    if desired_N < 2:
                        print(f"⚠️ Skipping FFT for '{trace.name}': resolution too fine for sample rate.")
                        continue
                    if N > desired_N:
                        y = y[:desired_N]
                        t = t[:desired_N]
                        N = desired_N
                else:
                    desired_N = int(min_duration * fs)
                    if desired_N < 2:
                        print(f"⚠️ Skipping FFT for '{trace.name}': signal too short for desired frequency window.")
                        continue
                    if N > desired_N:
                        y = y[:desired_N]
                        t = t[:desired_N]
                        N = desired_N

                window = np.ones(N) if window_type in ['raw', 'rect'] else get_window(window_type, N)
                y_windowed = y * window
                scale = np.sum(window) / N

                Npad = pad_factor * N
                if Npad == 0:
                    print(f"⚠️ Skipping FFT for '{trace.name}': FFT length is zero.")
                    continue

                yf = np.fft.rfft(y_windowed, n=Npad)
                xf = np.fft.rfftfreq(Npad, dt)
                magnitude = np.abs(yf) / (N / 2) / scale
                magnitude[0] /= 2

                if mag_mode == 'power':
                    magnitude = magnitude ** 2
                elif mag_mode == 'db':
                    magnitude = 20 * np.log10(magnitude + 1e-12)

                mask = (xf >= fmin) & (xf <= fmax)
                xf = xf[mask]
                magnitude = magnitude[mask]

                if len(xf) == 0 or len(magnitude) == 0:
                    print(f"⚠️ Skipping FFT for '{trace.name}': no data in frequency range.")
                    continue

                if len(xf) > max_points:
                    step = len(xf) // max_points
                    xf = xf[::step]
                    magnitude = magnitude[::step]

                color = color_cycle[color_index % len(color_cycle)]
                fig.add_trace(go.Scatter(
                    x=xf,
                    y=magnitude,
                    mode='lines',
                    name=f"{trace.name} FFT",
                    hovertemplate=
                        "<b>Freq</b>: %{x} Hz<br>" +
                        f"<b>Mag</b>: %{{y:.2f}} {mag_mode}<extra></extra>",
                    line=dict(color=color, width=2)
                ), row=idx+1, col=1)

                peak_index = np.argmax(magnitude)
                f_peak = xf[peak_index]
                fig.add_shape(
                    type="line",
                    x0=f_peak, x1=f_peak,
                    y0=min(magnitude), y1=max(magnitude),
                    line=dict(color=color, width=1, dash='dot'),
                    row=idx+1, col=1
                )
                fig.add_annotation(
                    text=f"{mn.tm(f_peak)}Hz",
                    x=f_peak,
                    y=max(magnitude),
                    showarrow=True,
                    font=dict(color=color),
                    xanchor='left',
                    row=idx+1, col=1
                )
                color_index += 1

        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=fig_height,
            margin=dict(t=40, b=20, l=40, r=20),
            paper_bgcolor=theme.background,
            plot_bgcolor=theme.background,
            font=dict(color=theme.font),
            hovermode='x unified'
        )
        fig.show()
