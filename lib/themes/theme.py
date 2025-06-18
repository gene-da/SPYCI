from dataclasses import dataclass, field
from typing import List

@dataclass
class PlotTheme:
    name: str
    background: str
    grid: str
    font: str
    accent: str
    colors: List[str] = field(default_factory=list)

    @staticmethod
    def tokyonight():
        return PlotTheme(
            name="TokyoNight",
            background="#1a1b26",
            grid="#3b4261",
            font="#c0caf5",
            accent="#7aa2f7",
            colors=[
                "#7aa2f7",  # Blue
                "#f7768e",  # Red
                "#bb9af7",  # Purple
                "#9ece6a",  # Green
                "#e0af68",  # Orange
                "#2ac3de",  # Cyan
                "#f38ba8",  # Pink
            ]
        )

    @staticmethod
    def ayu_mirage():
        return PlotTheme(
            name="Ayu Mirage",
            background="#1f2430",
            grid="#2e3440",
            font="#cbccc6",
            accent="#ffcc66",
            colors=[
                "#ffcc66",  # Yellow
                "#f07178",  # Red
                "#c3a6ff",  # Purple
                "#aad94c",  # Green
                "#86e1fc",  # Blue
                "#5ccfe6",  # Cyan
                "#e6b450",  # Orange
            ]
        )
