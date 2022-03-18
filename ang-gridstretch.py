#!/usr/bin/env python3

import argparse

import numpy as np
import scipy
import damask

from pathlib import Path


def bounding_box(cloud,return_size=False):
    mn,mx = np.min(cloud,axis=0),np.max(cloud,axis=0)
    return np.vstack((mn,mx)),mx-mn if return_size else np.vstack((mn,mx))

def interp(p,node_values):
    return np.einsum('i...,ij...->j...' if len(p.shape)>1 else 'i...,i...',
                     node_values,
                     np.array([
                               (1-p[...,0])*(1-p[...,1]),
                               (  p[...,0])*(1-p[...,1]),
                               (  p[...,0])*(  p[...,1]),
                               (1-p[...,0])*(  p[...,1]),
                              ]))

def match(needle,haystack):
    for i,item in enumerate(haystack):
        if item.startswith(needle):
            return i
    return None


grid_info = dict(square = {'label': 'SqrGrid',
                           'yfactor': 1,
                           'xshift': 0,
                           'threshold': 0},
                 hex = {'label': 'HexGrid',
                        'yfactor': np.sqrt(3)/2,
                        'xshift': 0.5,
                        'threshold': 1},
                )

parser = argparse.ArgumentParser(description='regrid and stretch orientation imaging maps')
parser.add_argument('--grid',default='hex',choices=['hex','square'],
                    help='output grid type')
parser.add_argument('--resolution',type=float,
                    metavar='float',
                    help='output grid resolution')
parser.add_argument('--stretch',nargs=8,type=float,
                    metavar='float',default=np.zeros((4,2)),
                    help='displacements (x,y) of the four orignal map corners')
parser.add_argument('file',
                    help='ANG file to regrid and stretch')

args = parser.parse_args()

file = Path(args.file)

ANG = damask.Table.load_ang(file)
cloud = ANG.get('pos')
bbox,diag = bounding_box(cloud,return_size=True)

cloud += interp((cloud-bbox[0])/diag,np.array(args.stretch).reshape((4,2)))
bbox,diag = bounding_box(cloud,return_size=True)

grid = np.stack(np.meshgrid(np.linspace(*bbox[:,0],np.round(diag[0]/args.resolution).astype(int)+1),
                            np.linspace(*bbox[:,1],np.round(diag[1]/args.resolution/grid_info[args.grid]['yfactor']).astype(int)+1)),
                axis=-1)

rows,cols = grid.shape[:2]
grid[1::2,:,0] += diag[0]/np.round(diag[0]/args.resolution)*grid_info[args.grid]["xshift"]
grid = grid.reshape(-1,2)[(1+np.arange(rows*cols))%(2*cols)>=grid_info[args.grid]["threshold"]]

for k,v in {'GRID: ':grid_info[args.grid]["label"],
            'NROWS: ':rows,
            'NCOLS_ODD: ':cols,
            'NCOLS_EVEN: ':cols-grid_info[args.grid]["threshold"]}.items():
    if m := match(k,ANG.comments):
        ANG.comments[m] = f'{k}{v}'
    else:
        ANG.comments.append(f'{k}{v}')

ANG[scipy.spatial.KDTree(cloud).query(grid)[1],:]\
   .set('pos',grid)\
   .save(file.with_stem(file.stem + '_stretched'),with_labels=False)
