import pytest
import time
import scipy.sparse
import math

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

other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


@pytest.mark.suitesparse
def test_mat_sddmm_locate_fused(samBench, ssname, cast, check_gold, report_stats, debug_sim, backpressure, depth, fill=0, KDIM=256):
    B_dirname = os.path.join(formatted_dir, ssname, "mat_sddmm")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    C_shape = (B_shape[0], KDIM)
    C_vals = np.arange(math.prod(C_shape)).tolist()

    D_shape = (KDIM, B_shape[1])
    D_vals = np.arange(math.prod(D_shape)).tolist()

    fiberlookup_Bi_25 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_19 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    repsiggen_k_11 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Bk_10 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))

    repsiggen_i_22 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Di_21 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Dk_14 = UncompressCrdRdScan(dim=D_shape[0], debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))

    repsiggen_j_16 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Cj_15 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_13 = UncompressCrdRdScan(dim=C_shape[1], debug=debug_sim, statistics=report_stats, back_en=backpressure,
                                            depth=int(depth))

    intersectk_12 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))

    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=len(B_crd0), fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    fiberwrite_X1_1 = CompressWrScan(seg_size=len(B_crd0) + 1, size=len(B_vals), fill=fill,
                                     debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_D_8 = Array(init_arr=D_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_6 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_4 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_3 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * len(B_vals), fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_D = [0, 'D']
    done = False
    time_cnt = 0

    temp = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    temp5 = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_25.set_in_ref(in_ref_B.pop(0), "")

        fiberlookup_Bj_19.set_in_ref(fiberlookup_Bi_25.out_ref(), fiberlookup_Bi_25)

        repsiggen_i_22.set_istream(fiberlookup_Bi_25.out_crd(), fiberlookup_Bi_25)

        if len(in_ref_D) > 0:
            repeat_Di_21.set_in_ref(in_ref_D.pop(0), "")
        repeat_Di_21.set_in_repsig(repsiggen_i_22.out_repsig(), repsiggen_i_22)

        fiberlookup_Dk_14.set_in_ref(fiberlookup_Bj_19.out_crd(), fiberlookup_Bj_19)

        repsiggen_j_16.set_istream(fiberlookup_Bj_19.out_crd(), fiberlookup_Bj_19)

        repeat_Cj_15.set_in_ref(fiberlookup_Bi_25.out_crd(), fiberlookup_Bi_25)
        repeat_Cj_15.set_in_repsig(repsiggen_j_16.out_repsig(), repsiggen_j_16)

        fiberlookup_Ck_13.set_in_ref(repeat_Cj_15.out_ref(), repeat_Cj_15)

        intersectk_12.set_in1(fiberlookup_Ck_13.out_ref(), fiberlookup_Ck_13.out_crd(), fiberlookup_Ck_13)
        intersectk_12.set_in2(fiberlookup_Dk_14.out_ref(), fiberlookup_Dk_14.out_crd(), fiberlookup_Dk_14)

        repsiggen_k_11.set_istream(intersectk_12.out_crd(), intersectk_12)

        arrayvals_C_7.set_load(intersectk_12.out_ref1(), intersectk_12)

        arrayvals_D_8.set_load(intersectk_12.out_ref2(), intersectk_12)

        repeat_Bk_10.set_in_ref(fiberlookup_Bj_19.out_ref(), fiberlookup_Bj_19)
        repeat_Bk_10.set_in_repsig(repsiggen_k_11.out_repsig(), repsiggen_k_11)

        arrayvals_B_6.set_load(repeat_Bk_10.out_ref(), repeat_Bk_10)

        mul_5.set_in1(arrayvals_B_6.out_val(), arrayvals_B_6)
        mul_5.set_in2(arrayvals_C_7.out_val(), arrayvals_C_7)

        mul_4.set_in1(mul_5.out_val(), mul_5)
        mul_4.set_in2(arrayvals_D_8.out_val(), arrayvals_D_8)

        reduce_3.set_in_val(mul_4.out_val(), mul_4)

        fiberwrite_Xvals_0.set_input(reduce_3.out_val(), reduce_3)

        fiberwrite_X0_2.set_input(fiberlookup_Bi_25.out_crd(), fiberlookup_Bi_25)

        fiberwrite_X1_1.set_input(fiberlookup_Bj_19.out_crd(), fiberlookup_Bj_19)

        fiberlookup_Bi_25.update()
        fiberlookup_Bj_19.update()
        repsiggen_i_22.update()
        repeat_Di_21.update()
        fiberlookup_Dk_14.update()
        repsiggen_j_16.update()
        repeat_Cj_15.update()
        fiberlookup_Ck_13.update()
        intersectk_12.update()
        repsiggen_k_11.update()
        arrayvals_C_7.update()
        arrayvals_D_8.update()
        repeat_Bk_10.update()
        arrayvals_B_6.update()
        mul_5.update()
        mul_4.update()
        reduce_3.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X0_2.update()
        fiberwrite_X1_1.update()

        done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    print("TOTAL TIME", time_cnt)

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
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_D_shape"] = D_shape

    extra_info["tensor_B/nnz"] = len(B_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)
    extra_info["tensor_D/nnz"] = len(D_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Di_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Di_21" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Cj_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cj_15" + "/" + k] = sample_dict[k]

    sample_dict = intersectk_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_12" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Bk_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bk_10" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_6" + "/" + k] = sample_dict[k]

    sample_dict = reduce_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_D_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_D_8" + "/" + k] = sample_dict[k]

    sample_dict = mul_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "/" + k] = sample_dict[k]

    sample_dict = mul_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_D_8" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_25" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_19" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Dk_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Dk_14" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_13" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_sddmm(ssname, debug_sim, cast, out_crds, out_segs, out_vals, "ss01", KDIM)
    samBench(bench, extra_info)
