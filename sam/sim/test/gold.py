import scipy.sparse
import scipy.io
import pytest
import os
import math
import torch
import numpy as np

from sam.onyx.generate_matrices import *
from sam.sim.src.base import *
from sam.sim.test.test import *

from sam.sim.test.test import check_point_tuple, remove_zeros, convert_point_tuple, convert_ndarr_point_tuple, \
    get_point_list, read_inputs
from sam.util import TnsFileLoader, round_sparse, ScipyTensorShifter, \
    SUITESPARSE_FORMATTED_PATH, SUITESPARSE_PATH, FROSTT_PATH, VALIDATION_OUTPUT_PATH, FROSTT_FORMATTED_PATH

cwd = os.getcwd()
ss_dir = SUITESPARSE_PATH
ss_formatted_dir = SUITESPARSE_FORMATTED_PATH
frostt_dir = FROSTT_PATH
validate_dir = VALIDATION_OUTPUT_PATH
tiled_output_path = os.getenv('TILED_OUTPUT_PATH', default=os.path.join(cwd, 'mode-formats'))


def check_gold_matmul_tiled(tile_crd_b, tile_crd_c, ssname, debug_sim, out_crds, out_segs, out_val, out_format="ss01"):
    # CSR
    gold_file_path = "out_" + str(tile_crd_b[0]) + "_" + str(tile_crd_b[1]) + "_" +\
        str(tile_crd_c[1]) + "_" + str(tile_crd_b[2]) + "_" + str(tile_crd_b[3]) +\
        "_" + str(tile_crd_c[3]) + ".mtx"
    gold_path = os.path.join(tiled_output_path, gold_file_path)
    # print(gold_path)
    if not os.path.exists(gold_path):
        if len(out_val) == 0:
            return
        if np.sum(np.asarray(out_val)) == 0:
            return
        else:
            print(out_val)
            assert False
    gold_nd = scipy.io.mmread(gold_path).toarray()
    transpose = out_format[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim and len(out_val) > 0:  # debug_sim:
        print("The  array is here")
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        # print("Dense Mat1:\n", B_scipy.toarray())
        # print("Dense Mat2:\n", C_scipy.toarray())
        # print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)
    if debug_sim:
        print(gold_tup, tile_crd_b, tile_crd_c)
        print(out_crds)
        print(out_segs)
        print(len(gold_tup))
        print(tile_crd_b, tile_crd_c)
        print(len(out_crds))
        print(len(out_segs))
        print(len(out_val))
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        if debug_sim:
            print("Out:", out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_matmul(ssname, debug_sim, cast, out_crds, out_segs, out_val, out_format="ss01"):
    # CSR
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    shifter = ScipyTensorShifter()

    B_scipy = B_tensor
    C_scipy = shifter.shiftLastMode(B_scipy).transpose()

    gold_nd = (B_scipy * C_scipy).toarray()
    transpose = out_format[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        if debug_sim:
            print("Out:", out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_elemmul(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    shifter = ScipyTensorShifter()
    B_scipy = B_tensor
    C_scipy = shifter.shiftLastMode(B_scipy)

    gold_nd = (B_scipy.multiply(C_scipy)).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_identity(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    B_scipy = B_tensor

    gold_nd = B_scipy.toarray()
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_elemadd(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    shifter = ScipyTensorShifter()
    B_scipy = B_tensor
    C_scipy = shifter.shiftLastMode(B_scipy)

    gold_nd = (B_scipy + C_scipy).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_vecmul_ji(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    return check_gold_mat_vecmul(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str)


def check_gold_mat_vecmul_ij(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    return check_gold_mat_vecmul(ssname, debug_sim, False, out_crds, out_segs, out_val, format_str)


def check_gold_mat_vecmul(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    c_dirname = os.path.join(ss_formatted_dir, ssname, "mat_vecmul")
    c_shape = B_tensor.shape[1]
    c0_crd_filename = os.path.join(c_dirname, "tensor_c_mode_0_crd")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "tensor_c_mode_vals")
    c_vals = read_inputs(c_vals_filename, float)

    B_scipy = B_tensor
    c_nd = np.zeros(c_shape)

    for i in range(len(c_crd0)):
        val = c_vals[i]
        crd = c_crd0[i]
        c_nd[crd] = val

    gold_nd = (B_scipy @ c_nd)
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Gold:", gold_tup)
        print()
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Vec2:\n", c_nd)
        print("Dense Gold:", gold_nd)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_sddmm(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str, KDIM=256):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    B_shape = B_tensor.shape
    C_shape = (B_shape[0], KDIM)
    C_vals = np.arange(math.prod(C_shape)).reshape(C_shape)

    D_shape = (KDIM, B_shape[1])
    D_vals = np.arange(math.prod(D_shape)).reshape(D_shape[::-1]).transpose()

    B_scipy = B_tensor

    gold_nd = (B_scipy.multiply(C_vals @ D_vals)).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_vals)
        print("Dense Mat3:\n", D_vals)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        if debug_sim:
            print("Out:", out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_residual(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    C_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in C_tensor.data]
        C_tensor = scipy.sparse.csr_matrix((data, C_tensor.indices, C_tensor.indptr), dtype=int)

    b_dirname = os.path.join(ss_formatted_dir, ssname, "mat_residual")
    b_shape = C_tensor.shape[0]
    b0_crd_filename = os.path.join(b_dirname, "tensor_b_mode_0_crd")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "tensor_b_mode_vals")
    b_vals = read_inputs(b_vals_filename, float)

    d_dirname = os.path.join(ss_formatted_dir, ssname, "mat_residual")
    d_shape = C_tensor.shape[1]
    d0_crd_filename = os.path.join(d_dirname, "tensor_d_mode_0_crd")
    d_crd0 = read_inputs(d0_crd_filename)

    d_vals_filename = os.path.join(d_dirname, "tensor_d_mode_vals")
    d_vals = read_inputs(d_vals_filename, float)

    C_scipy = C_tensor
    b_nd = np.zeros(b_shape)
    d_nd = np.zeros(d_shape)

    for i in range(len(b_crd0)):
        val = b_vals[i]
        crd = b_crd0[i]
        b_nd[crd] = val

    for i in range(len(d_crd0)):
        val = d_vals[i]
        crd = d_crd0[i]
        d_nd[crd] = val

    gold_nd = b_nd - (C_scipy @ d_nd)
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Vec1:\n", b_nd)
        print("Dense Mat1:\n", C_scipy.toarray())
        print("Dense Vec2:\n", d_nd)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_mattransmul(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    C_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in C_tensor.data]
        C_tensor = scipy.sparse.csr_matrix((data, C_tensor.indices, C_tensor.indptr), dtype=int)

    d_dirname = os.path.join(ss_formatted_dir, ssname, "mat_mattransmul")
    d_shape = C_tensor.shape[0]
    d0_crd_filename = os.path.join(d_dirname, "tensor_d_mode_0_crd")
    d_crd0 = read_inputs(d0_crd_filename)

    d_vals_filename = os.path.join(d_dirname, "tensor_d_mode_vals")
    d_vals = read_inputs(d_vals_filename, float)

    f_dirname = os.path.join(ss_formatted_dir, ssname, "mat_mattransmul")
    f_shape = C_tensor.shape[1]
    f0_crd_filename = os.path.join(f_dirname, "tensor_f_mode_0_crd")
    f_crd0 = read_inputs(f0_crd_filename)

    f_vals_filename = os.path.join(f_dirname, "tensor_f_mode_vals")
    f_vals = read_inputs(f_vals_filename, float)

    b = 2
    e = 2

    C_scipy = C_tensor
    d_nd = np.zeros(d_shape)
    f_nd = np.zeros(f_shape)

    for i in range(len(d_crd0)):
        val = d_vals[i]
        crd = d_crd0[i]
        d_nd[crd] = val

    for i in range(len(f_crd0)):
        val = f_vals[i]
        crd = f_crd0[i]
        f_nd[crd] = val

    gold_nd = b * C_scipy.T @ d_nd + e * f_nd
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Vec1:\n", d_nd)
        print("Dense Mat1:\n", C_scipy.transpose().toarray())
        print("Dense Vec2:\n", f_nd)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_elemadd3(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    shifter = ScipyTensorShifter()
    B_scipy = B_tensor
    C_scipy = shifter.shiftLastMode(B_scipy)
    D_scipy = shifter.shiftLastMode(C_scipy)

    gold_nd = (B_scipy + C_scipy + D_scipy).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor3_elemadd(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-plus2-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor3_ttv(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-ttv-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
    print(out_segs)
    print(out_crds)
    print(out_val)
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor3_ttm(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-ttm-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
    print("GOLD:", gold_tup)
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor3_innerprod(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    if frosttname == "fb1k":
        assert out_val == [1066.0]
    else:
        assert False, "Gold not entered yet"


def check_gold_tensor3_mttkrp(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-mttkrp-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
    print("GOLD:", gold_tup)
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def remove_items(test_list, item):
 
    # using filter() + __ne__ to perform the task
    res = list(filter((item).__ne__, test_list))
 
    return res




def check_gold_tensor3_linear(frosttname, debug_sim, out_crds, out_segs, out_vals, format_str):
    formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    B_dirname = os.path.join(formatted_dir, frosttname, "orig", "ss01")
    C_dirname = os.path.join(formatted_dir, frosttname, "other", "sss021")
    D_dirname = os.path.join(formatted_dir, frosttname, "other", "s0")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)
    D_shape_filename = os.path.join(D_dirname, "D_shape.txt")
    D_shape = read_inputs(D_shape_filename)
    B_tens = get_tensor_from_files(name="B", files_dir=B_dirname, shape=B_shape, base=10, early_terminate='x')
    C_tens = get_tensor_from_files(name="C", files_dir=C_dirname, shape=C_shape, base=10, early_terminate='x')
    D_tens = get_tensor_from_files(name="D", files_dir=D_dirname, shape=D_shape, base=10, early_terminate='x')
    # B_dirname_trans = os.path.join(formatted_dir, frosttname, "orig", "ssss0213")
    # C_dirname_trans = os.path.join(formatted_dir, frosttname, "other", "sss021")
    # mode = (0,2,1)
    # B_tens.transpose_tensor(mode)
    # C_tens.transpose_tensor(mode)
    # B_tens.set_dump_dir(B_dirname_trans)
    # C_tens.set_dump_dir(C_dirname_trans)

    # B_tens.dump_outputs(format='CSF')
    # C_tens.dump_outputs(format='CSF')

    # pytest.set_trace()

    # pytest.set_trace()
    B_ref = torch.from_numpy(B_tens.get_matrix())
    C_ref = torch.from_numpy(C_tens.get_matrix())
    D_ref = torch.from_numpy(D_tens.get_matrix())
    # D_ref = torch.unsqueeze(D_ref, 0).unsqueeze(2)
    # D_ref = torch.broadcast_to(D_ref, (2, 10, 10))
    print(D_ref.shape)

    pytest.set_trace()

    gold_ref = torch.einsum('jl, ilk->ijk', B_ref, C_ref)
    gold_tup = convert_ndarr_point_tuple(gold_ref.numpy())
    print("Before add: ", gold_tup)
    gold_ref = D_ref + gold_ref
    gold_ref = gold_ref.numpy()
    # gold_ref = torch.einsum('jl, ilk->ijk', B_ref, C_ref).numpy()

    mat_g = MatrixGenerator("gold", shape=gold_ref.shape, sparsity=0.1, format='CSF', dump_dir='test', tensor=gold_ref)
    mat_g.dump_outputs(format='CSF')
    gold_tup = convert_ndarr_point_tuple(gold_ref)
    print("After add: ", gold_tup)
    pytest.set_trace()
    # mg = create_matrix_from_point_list("gold", gold_tup, gold_ref.shape)
    # print(mg.get_matrix())
    print("Out crds:", out_crds)
    print()
    print("Out segs:", out_segs)
    print()
    print("Out vals:", out_vals)
    print(len(out_vals))
    print("sizes:", [len(arr) for arr in out_crds])
    print("sizes:", [len(arr) for arr in out_segs])
    print(gold_ref.shape)
    pytest.set_trace()

    if debug_sim:
        print("Out crds:", out_crds)
        print("Out segs:", out_segs)
        print("Out vals:", out_vals)
        print("Dense Gold:", gold_ref)
        print("Gold:", gold_tup)

    if not out_vals:
        assert out_vals == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_vals])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_vals))
        out_tup = remove_zeros(out_tup)
        print("ref:", out_tup)
        print("gold:", gold_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor4_multiply(frosttname, debug_sim, out_crds, out_segs, out_vals, format_str):
    formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    B_dirname = os.path.join(formatted_dir, frosttname, "orig", "ssss0123")
    C_dirname = os.path.join(formatted_dir, frosttname, "other", "ssss0123")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)
    B_tens = get_tensor_from_files(name="B", files_dir=B_dirname, shape=B_shape, base=10, early_terminate='x')
    C_tens = get_tensor_from_files(name="C", files_dir=C_dirname, shape=C_shape, base=10, early_terminate='x')
    # B_dirname_trans = os.path.join(formatted_dir, frosttname, "orig", "ssss0213")
    # C_dirname_trans = os.path.join(formatted_dir, frosttname, "other", "sss021")
    # mode = (0,2,1)
    # B_tens.transpose_tensor(mode)
    # C_tens.transpose_tensor(mode)
    # B_tens.set_dump_dir(B_dirname_trans)
    # C_tens.set_dump_dir(C_dirname_trans)

    # B_tens.dump_outputs(format='CSF')
    # C_tens.dump_outputs(format='CSF')

    # pytest.set_trace()

    # pytest.set_trace()
    B_ref = torch.from_numpy(B_tens.get_matrix())
    C_ref = torch.from_numpy(C_tens.get_matrix())

    gold_ref = torch.einsum('ikjm, iljm->ijkl', B_ref, C_ref).numpy()

    mat_g = MatrixGenerator("gold", shape=gold_ref.shape, sparsity=0.1, format='CSF', dump_dir='test', tensor=gold_ref)
    mat_g.dump_outputs(format='CSF')
    gold_tup = convert_ndarr_point_tuple(gold_ref)
    # mg = create_matrix_from_point_list("gold", gold_tup, gold_ref.shape)
    # print(mg.get_matrix())
    print("Out crds:", out_crds)
    print()
    print("Out segs:", out_segs)
    print()
    print("Out vals:", out_vals)
    print(len(out_vals))
    print("sizes:", [len(arr) for arr in out_crds])
    print("sizes:", [len(arr) for arr in out_segs])
    print(gold_ref.shape)
    # pytest.set_trace()

    if debug_sim:
        print("Out crds:", out_crds)
        print("Out segs:", out_segs)
        print("Out vals:", out_vals)
        print("Dense Gold:", gold_ref)
        print("Gold:", gold_tup)

    if not out_vals:
        assert out_vals == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_vals])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_vals))
        out_tup = remove_zeros(out_tup)
        print("ref:", out_tup)
        print("gold:", gold_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor4_multiply2(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, format_str):
    B_dirname = os.path.join(FROSTT_FORMATTED_PATH, frosttname, "tensor4_multiply2")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)
    C_dirname = os.path.join(FROSTT_FORMATTED_PATH, frosttname, "tensor4_multiply2")
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)

    B_tens = get_tensor_from_files(name="B", files_dir=B_dirname, shape=B_shape, base=10, early_terminate='x')
    C_tens = get_tensor_from_files(name="C", files_dir=C_dirname, shape=C_shape, base=10, early_terminate='x')
    # B_dirname_trans = os.path.join(formatted_dir, frosttname, "orig", "ssss0213")
    # C_dirname_trans = os.path.join(formatted_dir, frosttname, "other", "ssss0231")
    # mode1 = (0,2,1,3)
    # mode2 = (0,2,3,1)
    # B_tens.transpose_tensor(mode1)
    # C_tens.transpose_tensor(mode2)
    # B_tens.set_dump_dir(B_dirname_trans)
    # C_tens.set_dump_dir(C_dirname_trans)

    # B_tens.dump_outputs(format='CSF')
    # C_tens.dump_outputs(format='CSF')

    # pytest.set_trace()

    # pytest.set_trace()
    print("B numpy shape:", B_tens.shape)
    print("C numpy shape:", C_tens.shape)

    B_ref = torch.from_numpy(B_tens.get_matrix())
    C_ref = torch.from_numpy(C_tens.get_matrix())

    B_ref = torch.permute(B_ref, (0, 2, 1, 3))
    C_ref = torch.permute(C_ref, (0, 3, 1, 2))
    
    print(B_ref.shape)
    print(C_ref.shape)
    pytest.set_trace()

    gold_ref = torch.einsum('ijkl, iljm->ikjm', B_ref, C_ref).numpy()

    mat_g = MatrixGenerator("gold", shape=gold_ref.shape, sparsity=0.1, format='CSF', dump_dir='test', tensor=gold_ref)
    mat_g.dump_outputs(format='CSF')
    gold_tup = convert_ndarr_point_tuple(gold_ref)
    # mg = create_matrix_from_point_list("gold", gold_tup, gold_ref.shape)
    # print(mg.get_matrix())
    print("Out crds:", out_crds)
    print()
    print("Out segs:", out_segs)
    print()
    print("Out vals:", out_vals)
    print(len(out_vals))
    print("sizes:", [len(arr) for arr in out_crds])
    print("sizes:", [len(arr) for arr in out_segs])
    print(gold_ref.shape)
    pytest.set_trace()

    if debug_sim:
        print("Out crds:", out_crds)
        print("Out segs:", out_segs)
        print("Out vals:", out_vals)
        print("Dense Gold:", gold_ref)
        print("Gold:", gold_tup)

    if not out_vals:
        assert out_vals == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_vals])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_vals))
        out_tup = remove_zeros(out_tup)
        print("ref:", out_tup)
        print("gold:", gold_tup)
        assert (check_point_tuple(out_tup, gold_tup))


# ---------------- OTHER CHECKS (TODO later) ---------------- #
def check_gold_tensor3_identity(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    pass


def check_gold_tensor3_relu(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    pass
