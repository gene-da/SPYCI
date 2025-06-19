from lib.utils.resonant import *
from lib.utils import metric_notation as mn
from tabulate import tabulate
from lib.utils.calcs_output import save_markdown_table as save
import numpy as np
from lib.utils import filter as fltr
from lib.utils.standard_comp import nearest_standard as ns


frequencies = [
    mn.fm('430k'), mn.fm('445k'), mn.fm('1000k'), mn.fm('1700k'), mn.fm('1445k')
]

inductors = [
    mn.fm('1u'), mn.fm('2u'), mn.fm('4u'), mn.fm('8u'),
    mn.fm('10u'), mn.fm('20u'), mn.fm('40u'), mn.fm('80u'), 
    mn.fm('100u'), mn.fm('200u'), mn.fm('400u'), mn.fm('800u')
]


table = [['Inductance (H)'] + [f'{mn.tm(f)}Hz' for f in frequencies]]

res = mn.fm('50')

# Loop through inductors and calculate capacitance for each frequency
for l in inductors:
    row = [mn.tm(l, 0)]
    for f in frequencies:
        c = ns(res_c(l, f))
        q = (q_of_coil(l, 50, f))
        row.append(f'C:{mn.tm(c, 1)}F Q:{q:.2f}')  # 2 significant digits for clarity
    table.append(row)

save('resonance_calcs', table)

freq1 = mn.fm('430k')
freq2 = mn.fm('445k')
freq3 = mn.fm('1000k')
freq4 = mn.fm('1700k')
freq5 = mn.fm('1445k')


data1 = [
    ['Resistance', 
     f'{mn.tm(freq1)}Hz',
     f'{mn.tm(freq2)}Hz',
     f'{mn.tm(freq3)}Hz',
     f'{mn.tm(freq4)}Hz',
     f'{mn.tm(freq5)}Hz']
]

for res in range(100, 510, 10):
    c1 = fltr.rcf_rescut(res, freq1)
    c2 = fltr.rcf_rescut(res, freq2)
    c3 = fltr.rcf_rescut(res, freq3)
    c4 = fltr.rcf_rescut(res, freq4)
    c5 = fltr.rcf_rescut(res, freq5)
    
    data1.append([f'{mn.tm(res)}Ω',
                  f'{mn.tm(c1)}F', 
                  f'{mn.tm(c2)}F',
                  f'{mn.tm(c3)}F',
                  f'{mn.tm(c4)}F',
                  f'{mn.tm(c5)}F'])
    
save('emitter_cutoffs_calcs', data1)

