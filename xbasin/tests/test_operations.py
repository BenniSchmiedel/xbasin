import xarray as xr
import xgcm
import numpy as np
import warnings
from xnemogcm import open_nemo_and_domain_cfg
import pytest

from xbasin.operations import Grid_ops
_metrics = {
    ("X",): ["e1t", "e1u", "e1v", "e1f"],  # X distances
    ("Y",): ["e2t", "e2u", "e2v", "e2f"],  # Y distances
    ("Z",): ["e3t_0", "e3u_0", "e3v_0", "e3f_0", "e3w_0"],  # Z distances
}

def _assert_same_position(grid_ops,data,position):
    check=grid_ops._matching_pos(data,position)
    if type(check) is list:
        assert all(check)
    else:
        assert check

def test_shift_position_to_T():

    #ds = open_nemo_and_domain_cfg(datadir='data')
    domcfg = xr.open_dataset("data/xnemogcm.domcfg_to.nc")
    nemo_ds = xr.open_dataset("data/xnemogcm.nemo.nc")
    grid = xgcm.Grid(domcfg,metrics=_metrics,periodic=False)
    grid_ops = Grid_ops(grid)

    u_fr = nemo_ds.uo
    v_fr = nemo_ds.vo
    w_fr = nemo_ds.woce

    u_3d_fr = [u_fr ,v_fr ,w_fr]

    #Test single variables
    u_to = grid_ops._shift_position(u_fr,output_position='T')
    v_to = grid_ops._shift_position(v_fr, output_position='T')
    w_to = grid_ops._shift_position(w_fr, output_position='T')

    u_3d_to = grid_ops._shift_position(u_3d_fr,output_position='T')

    #grid_ops._matching_pos([u_to,v_to,w_to,u_3d_to],'T')
    _assert_same_position(grid_ops,[u_to,v_to,w_to],'T')

if __name__ == "__main__":
    pass