import pytest
import time
import scipy.sparse
from sam.sim.src.compression import ValDropper
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2, Divide2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult, Softmax
from sam.sim.src.crd_masker import Tril, Dropout
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce, MaxReduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv
cwd = os.getcwd()
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor4_softmax(samBench, frosttname, cast, check_gold, debug_sim, report_stats, fill=0):
    # test_name = "tensor4_mult2_ijklm"
    # test_name = "tensor4_softmax"
    test_name = "tensor4_softmax_large"
    B_dirname = os.path.join(formatted_dir, frosttname, test_name)
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

    B2_seg_filename = os.path.join(B_dirname, "tensor_B_mode_2_seg")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "tensor_B_mode_2_crd")
    B_crd2 = read_inputs(B2_crd_filename)

    B3_seg_filename = os.path.join(B_dirname, "tensor_B_mode_3_seg")
    B_seg3 = read_inputs(B3_seg_filename)
    B3_crd_filename = os.path.join(B_dirname, "tensor_B_mode_3_crd")
    B_crd3 = read_inputs(B3_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    out_crds = []
    out_segs = []
    out_vals = []
    check_gold_tensor4_softmax(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, test_name)

    fiberlookup_Bi_7 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_6 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_5 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bl_6 = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats)
    fiberlookup_dense_i = UncompressCrdRdScan(dim=B_shape[0], debug=debug_sim)
    fiberlookup_dense_j = UncompressCrdRdScan(dim=B_shape[1], debug=debug_sim)
    fiberlookup_dense_k = UncompressCrdRdScan(dim=B_shape[2], debug=debug_sim)
    fiberlookup_dense_l = UncompressCrdRdScan(dim=B_shape[3], debug=debug_sim)
    unionl_42 = Union2(debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X3_0 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] * B_shape[2] * B_shape[3], size=B_shape[0] * B_shape[1] * B_shape[2] * B_shape[3], fill=fill, debug=debug_sim, statistics=report_stats)
    arrayvals_B_4 = Array(init_arr=B_vals, datatype=float, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * B_shape[2] * B_shape[3], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_l_13 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bl_12 = Repeat(debug=debug_sim, statistics=report_stats)
    repsiggen_l1_13 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bl1_12 = Repeat(debug=debug_sim, statistics=report_stats)
    # Using this paper as baseline: 17 clock cycles -> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7309545
    exp_1 = Exp(in2=0, delay=1, debug=debug_sim, statistics=report_stats)
    reduce_5 = Reduce(debug=debug_sim, statistics=report_stats)
    max_reduce_5 = MaxReduce(debug=debug_sim, statistics=report_stats)
    drop_9 = CrdDrop(debug=debug_sim)
    div_6 = Divide2(debug=debug_sim, statistics=report_stats)
    add_10 = Add2(debug=debug_sim, neg2=True, statistics=report_stats)
    softmax = Softmax(debug=debug_sim)
    comp_drop_1 = ValDropper(debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    in_ref_B_dense = [0, 'D']
    done = False
    time_cnt = 0

    # print("B seg1", B_seg1)
    # print("B seg2", B_seg2)
    # print("B seg3", B_seg3)

    # pytest.set_trace()

    out_debug = []

    div_in = []
    div1_in = []
    div_out = []
    div1_out = []

    repeater = []
    reducer = []
    fiber_crd = []
    repsig = []

    out_l_crd = []

    test = []

    count = 0
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_7.set_in_ref(in_ref_B.pop(0))
        # fiberwrite_X0_3.set_input(fiberlookup_Bi_7.out_crd())
        fiberlookup_Bj_6.set_in_ref(fiberlookup_Bi_7.out_ref())
        # fiberwrite_X1_2.set_input(fiberlookup_Bj_6.out_crd())
        fiberlookup_Bk_5.set_in_ref(fiberlookup_Bj_6.out_ref())
        fiberlookup_Bl_6.set_in_ref(fiberlookup_Bk_5.out_ref())
        # fiberwrite_X2_1.set_input(fiberlookup_Bk_5.out_crd())
        # fiberwrite_X3_0.set_input(fiberlookup_Bl_6.out_crd())

        # if len(in_ref_B_dense) > 0:
        #     fiberlookup_dense_i.set_in_ref(in_ref_B_dense.pop(0))
        # fiberlookup_dense_j.set_in_ref(fiberlookup_dense_i.out_ref())
        # fiberlookup_dense_k.set_in_ref(fiberlookup_dense_j.out_ref())
        # fiberlookup_dense_l.set_in_ref(fiberlookup_dense_k.out_ref())

        # FIXME: Need to fix union below to get dense softmax,
        # Currently this ignores rows with only zeros which should output 1/(# rows)
        # unionl_42.set_in1(fiberlookup_Bl_6.out_ref(), fiberlookup_Bl_6.out_crd())
        # unionl_42.set_in2(fiberlookup_dense_l.out_ref(), fiberlookup_dense_l.out_crd())
        # arrayvals_B_4.set_load(unionl_42.out_ref1())

        # out_l_crd.append(fiberlookup_dense_l.out_crd())
        # out_l_crd.append(fiberlookup_dense_l.out_ref())
        # test.append(fiberlookup_Bl_6.out_ref())


        arrayvals_B_4.set_load(fiberlookup_Bl_6.out_ref())

        max_reduce_5.set_in_val(arrayvals_B_4.out_load())
        repsiggen_l1_13.set_istream(fiberlookup_Bl_6.out_ref())
        # repsiggen_l1_13.set_istream(unionl_42.out_ref1())
        # repsiggen_l1_13.set_istream(arrayvals_B_4.out_val())
        repeat_Bl1_12.set_in_ref(max_reduce_5.out_val())
        repeat_Bl1_12.set_in_repsig(repsiggen_l1_13.out_repsig())
        add_10.set_in1(arrayvals_B_4.out_load())
        add_10.set_in2(repeat_Bl1_12.out_ref())

        # div_in.append(arrayvals_B_4.out_val())
        div_in.append(max_reduce_5.out_val())
        div1_in.append(repeat_Bl1_12.out_ref())

        print("first:", remove_emptystr(div_in))
        print("second:", remove_emptystr(div1_in))
        
        exp_1.set_in1(add_10.out_val())
        test.append(arrayvals_B_4.out_val())
        # print("values:", remove_emptystr(test))
        reduce_5.set_in_val(exp_1.out_val())
        # repsiggen_l_13.set_istream(fiberlookup_Bl_6.out_ref())
        repeat_Bl_12.set_in_ref(reduce_5.out_val())
        repeat_Bl_12.set_in_repsig(repsiggen_l1_13.out_repsig())
        div_6.set_in1(exp_1.out_val())
        div_6.set_in2(repeat_Bl_12.out_ref())
        out_l_crd.append(div_6.out_val())
        # print("ref2:", remove_emptystr(out_l_crd))

        comp_drop_1.set_val(div_6.out_val())
        comp_drop_1.set_crd(fiberlookup_Bl_6.out_crd())
        comp_drop_1.set_ref(fiberlookup_Bl_6.out_ref())
        # comp_drop_1.set_crd(unionl_42.out_crd())
        # comp_drop_1.set_ref(unionl_42.out_ref1())

        # softmax.set_val(arrayvals_B_4.out_load())
        # softmax.set_inner_ref(fiberlookup_Bl_6.out_ref())

        # out_debug.append(softmax.out_val())
        out_debug.append(comp_drop_1.out_crd())

        # print("Val: ", remove_emptystr(out_debug))

        # fiber_crd.append(fiberlookup_Bl_6.out_crd())
        reducer.append(max_reduce_5.out_val())
        # repeater.append(repeat_Bl1_12.out_ref())
        repsig.append(add_10.out_val())

        # print("crd:", remove_emptystr(fiber_crd))
        # print('=' * 100)
        # print("Reduce:", remove_emptystr(reducer))
        # print("Repeater:", remove_emptystr(repeater))
        # print()
        #print("Repsig:", remove_emptystr(repsig))

        # div1_in.append(arrayvals_B_4.out_load())
        # out_debug.append(softmax.out_val())
        # print("softmax", remove_emptystr(out_debug))
        # print("array vals:", remove_emptystr(div1_in))
        # print()
        # print("div1 in", remove_emptystr(div1_in))
        # print()
        # print("div out", remove_emptystr(out_debug))
        # fiberwrite_Xvals_0.set_input(div_6.out_val())
        fiberwrite_Xvals_0.set_input(comp_drop_1.out_val())
        # fiberwrite_Xvals_0.set_input(softmax.out_val())
        fiberwrite_X0_3.set_input(fiberlookup_Bi_7.out_crd())
        fiberwrite_X1_2.set_input(fiberlookup_Bj_6.out_crd())
        fiberwrite_X2_1.set_input(fiberlookup_Bk_5.out_crd())
        # fiberwrite_X3_0.set_input(fiberlookup_Bl_6.out_crd())
        fiberwrite_X3_0.set_input(comp_drop_1.out_crd())

        fiberlookup_Bi_7.update()
        fiberlookup_Bj_6.update()
        fiberlookup_Bk_5.update()
        fiberlookup_Bl_6.update()
        # fiberlookup_dense_i.update()
        # fiberlookup_dense_j.update()
        # fiberlookup_dense_k.update()
        # fiberlookup_dense_l.update()
        # unionl_42.update()
        arrayvals_B_4.update()

        softmax.update()

        max_reduce_5.update()
        repsiggen_l1_13.update()
        repeat_Bl1_12.update()
        add_10.update()
        exp_1.update()
        reduce_5.update()
        # arrayvals_B_10.update()
        repsiggen_l_13.update()
        repeat_Bl_12.update()
        div_6.update()

        comp_drop_1.update()

        fiberwrite_X0_3.update()
        fiberwrite_X1_2.update()
        fiberwrite_X2_1.update()
        fiberwrite_X3_0.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and fiberwrite_Xvals_0.out_done()

        # print(done)
        # if done_:
            # count += 1
        # done = False
        # if count == 4:
        #     done = True
        # done = exp_1.out_done()
        # time_cnt += 1 if exp_1.get_computed() else 6
        exp_1.set_computed()
        time_cnt += 1 
        if time_cnt % 100000 == 0:
            print("Cycle: ", time_cnt)

    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X2_1.autosize()
    fiberwrite_X3_0.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X2_1.get_arr(), fiberwrite_X3_0.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X2_1.get_seg_arr(), fiberwrite_X3_0.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print("segs:", [len(out_seg) for out_seg in out_segs], out_segs)
    print("crds:", [len(out_crd) for out_crd in out_crds], out_crds)
    print("out_vals:", out_vals)
    print("# cycles: ", time_cnt)

    # pytest.set_trace()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    # extra_info["dataset"] = 
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    sample_dict = fiberlookup_Bi_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_7" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_softmax(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, test_name)
    samBench(bench, extra_info)