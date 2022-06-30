import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_mat_elemadd3_FINAL(samBench, ssname, check_gold, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss01")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, ssname, "shift", "ss01")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    D_shape = C_shape

    D_seg0 = copy.deepcopy(C_seg0)
    D_crd0 = copy.deepcopy(C_crd0)

    D_seg1 = copy.deepcopy(C_seg1)
    D_crd1 = copy.deepcopy(C_crd1)
    # Shift by one again and sort
    D_crd1 = [x + 1 if (x + 1) < D_shape[1] else 0 for x in D_crd1]
    for i in range(len(D_seg1) - 1):
        start = D_seg1[i]
        end = D_seg1[i + 1]
        D_crd1[start:end] = sorted(D_crd1[start:end])

    D_vals = copy.deepcopy(C_vals)

    fiberlookup_Bi_13 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberlookup_Ci_14 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    fiberlookup_Di_15 = CompressedCrdRdScan(crd_arr=D_crd0, seg_arr=D_seg0, debug=debug_sim)
    unioni1_12 = Union2(debug=debug_sim)
    unioni2_12 = Union2(debug=debug_sim)
    unioni3_12 = Union2(debug=debug_sim)
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=3 * len(B_crd0), fill=fill, debug=debug_sim)
    fiberlookup_Bj_9 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberlookup_Cj_10 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    fiberlookup_Dj_11 = CompressedCrdRdScan(crd_arr=D_crd1, seg_arr=D_seg1, debug=debug_sim)
    unionj1_8 = Union2(debug=debug_sim)
    unionj2_8 = Union2(debug=debug_sim)
    unionj3_8 = Union2(debug=debug_sim)
    fiberwrite_X1_1 = CompressWrScan(seg_size=3 * len(B_crd0) + 1, size=3 * len(B_vals), fill=fill, debug=debug_sim)
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim)
    arrayvals_D_7 = Array(init_arr=D_vals, debug=debug_sim)
    add_4 = Add2(debug=debug_sim)
    add_3 = Add2(debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=3 * len(B_vals), fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_D = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_13.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_13.update()

        if len(in_ref_C) > 0:
            fiberlookup_Ci_14.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ci_14.update()

        if len(in_ref_D) > 0:
            fiberlookup_Di_15.set_in_ref(in_ref_D.pop(0))
        fiberlookup_Di_15.update()

        unioni1_12.set_in1(fiberlookup_Bi_13.out_ref(), fiberlookup_Bi_13.out_crd())
        unioni1_12.set_in2(fiberlookup_Ci_14.out_ref(), fiberlookup_Ci_14.out_crd())
        unioni1_12.update()

        unioni2_12.set_in1(fiberlookup_Di_15.out_ref(), fiberlookup_Di_15.out_crd())
        unioni2_12.set_in2(unioni1_12.out_ref1(), unioni1_12.out_crd())
        unioni2_12.update()

        unioni3_12.set_in1(fiberlookup_Di_15.out_ref(), fiberlookup_Di_15.out_crd())
        unioni3_12.set_in2(unioni1_12.out_ref2(), unioni1_12.out_crd())
        unioni3_12.update()

        fiberwrite_X0_2.set_input(unioni2_12.out_crd())
        fiberwrite_X0_2.update()

        fiberlookup_Bj_9.set_in_ref(unioni2_12.out_ref2())
        fiberlookup_Bj_9.update()

        fiberlookup_Cj_10.set_in_ref(unioni3_12.out_ref2())
        fiberlookup_Cj_10.update()

        fiberlookup_Dj_11.set_in_ref(unioni3_12.out_ref1())
        fiberlookup_Dj_11.update()

        unionj1_8.set_in1(fiberlookup_Bj_9.out_ref(), fiberlookup_Bj_9.out_crd())
        unionj1_8.set_in2(fiberlookup_Cj_10.out_ref(), fiberlookup_Cj_10.out_crd())
        unionj1_8.update()

        unionj2_8.set_in1(fiberlookup_Dj_11.out_ref(), fiberlookup_Dj_11.out_crd())
        unionj2_8.set_in2(unionj1_8.out_ref1(), unionj1_8.out_crd())
        unionj2_8.update()

        unionj3_8.set_in1(fiberlookup_Dj_11.out_ref(), fiberlookup_Dj_11.out_crd())
        unionj3_8.set_in2(unionj1_8.out_ref2(), unionj1_8.out_crd())
        unionj3_8.update()

        fiberwrite_X1_1.set_input(unionj3_8.out_crd())
        fiberwrite_X1_1.update()

        arrayvals_B_5.set_load(unionj2_8.out_ref2())
        arrayvals_B_5.update()

        arrayvals_C_6.set_load(unionj3_8.out_ref2())
        arrayvals_C_6.update()

        arrayvals_D_7.set_load(unionj3_8.out_ref1())
        arrayvals_D_7.update()

        add_4.set_in1(arrayvals_B_5.out_val())
        add_4.set_in2(arrayvals_C_6.out_val())
        add_4.update()

        add_3.set_in1(add_4.out_val())
        add_3.set_in2(arrayvals_D_7.out_val())
        add_3.update()

        fiberwrite_Xvals_0.set_input(add_3.out_val())
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1
        if time_cnt % 1000000 == 0:
            print("TIME:", time_cnt)
            print("DONE:", fiberwrite_X0_2.out_done(), fiberwrite_X1_1.out_done(), fiberwrite_Xvals_0.out_done(),
                  unioni1_12.out_done(), unioni2_12.out_done(), unioni3_12.out_done(),
                  unionj1_8.out_done(), unionj2_8.out_done(), unionj3_8.out_done(),
                  fiberlookup_Bj_9.out_done(), fiberlookup_Cj_10.out_done(), fiberlookup_Dj_11.out_done())
    print("TOTAL TIME:", time_cnt)

    fiberwrite_X0_2.autosize()
    fiberwrite_X1_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
    out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["expression"] = "elemadd3"
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_D_shape"] = D_shape
    extra_info["suitesparse/nnz"] = len(B_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)
    extra_info["tensor_D/nnz"] = len(D_vals)
    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = unioni1_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni1_12" + "/" + k] = sample_dict[k]
    sample_dict = unioni2_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni2_12" + "/" + k] = sample_dict[k]
    sample_dict = unioni3_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni3_12" + "/" + k] = sample_dict[k]

    sample_dict = unionj1_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["unionj1_8" + "/" + k] = sample_dict[k]
    sample_dict = unioni2_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["unionj2_8" + "/" + k] = sample_dict[k]
    sample_dict = unioni3_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["unionj3_8" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_D_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_D_7" + "/" + k] = sample_dict[k]

    sample_dict = add_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["add_4" + "/" + k] = sample_dict[k]

    sample_dict = add_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["add_3" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_elemadd3(ssname, debug_sim, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
