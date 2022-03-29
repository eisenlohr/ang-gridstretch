#!/usr/bin/env python3

import argparse

import numpy as np
import pyvista as pv
import damask

from pathlib import Path

parser = argparse.ArgumentParser(description='transform orientation imaging map in ANG format (TSL/EDAX) to VTP.')
parser.add_argument('--ipf',nargs=3,default=[0,0,1],type=float,metavar='float',
                    help='Sample direction of inverse pole figure')
parser.add_argument('file',
                    help='ANG file to transform to VTP')

args = parser.parse_args()
file = Path(args.file)

t = damask.Table.load_ang(file)

mesh = pv.PolyData(np.column_stack((t.get('pos'),np.zeros(len(t)))))
mesh['IPF'] = damask.Orientation.from_Euler_angles(phi=t.get('eu'),family='cubic').IPF_color(args.ipf)
for l in set(t.labels) - {'pos'}:
    mesh[l] = t.get(l)

mesh.save(file.with_suffix('.vtp'))
