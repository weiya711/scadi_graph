import argparse
import os
import shutil
import scipy.sparse
import numpy as np
import sys

from pathlib import Path
from util import parse_taco_format

from util import FormatWriter, SuiteSparseTensor, InputCacheSuiteSparse
# custom_path = '/nobackup/jadivara/sam/sam/util.py'
# sys.path.append(custom_path)
# from  import SUITESPARSE_FORMATTED_PATH, ScipyTensorShifter

cwd = os.getcwd()

formats = ["sss012", "ss01", "dss", "dds", "ddd", "dsd", "sdd", "sds", "ssd"]

parser = argparse.ArgumentParser(description="Process some Frostt tensors into per-level datastructures")
parser.add_argument('-n', '--name', metavar='fname', type=str, action='store',
                    help='tensor name to run format conversion on one frostt tensor')
parser.add_argument('-f', '--format', metavar='fformat', type=str, action='store',
                    help='The format that the tensor should be converted to')
parser.add_argument('-i', '--int', action='store_false', default=True, help='Safe sparsity cast to int for values')
parser.add_argument('-s', '--shift', action='store_false', default=True, help='Also format shifted tensor')
parser.add_argument('-o', '--other', action='store_true', default=False, help='Format other tensor')
parser.add_argument('-ss', '--suitesparse', action='store_true', default=False, help='Format suitesparse other tensor')
parser.add_argument('-hw', '--hw', action='store_true', default=False,
                    help='Format filenames as in AHA SCGRA <tensor_<name>_mode_<n|type>')
parser.add_argument('-np', '--numpy', action='store_true', default=False, help='Format numpy tensors')
parser.add_argument('-b', '--bench', type=str, default=None, help='Name of benchmark')

args = parser.parse_args()
if args.other:
    if args.suitesparse:
        outdir_name = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    else:
        outdir_name = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    taco_format_dirname = os.getenv('TACO_TENSOR_PATH')
    if taco_format_dirname is None:
        print("Please set the TACO_TENSOR_PATH environment variable")
        exit()
    taco_format_dirname = os.path.join(taco_format_dirname, "other-formatted-taco")
else:
    outdir_name = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    taco_format_dirname = os.getenv('FROSTT_FORMATTED_TACO_PATH')
    if taco_format_dirname is None:
        print("Please set the FROSTT_FORMATTED_TACO_PATH environment variable")
        exit()

out_path = Path(outdir_name)
out_path.mkdir(parents=True, exist_ok=True)

if args.name is None:
    print("Please enter a tensor name")
    exit()

#breakpoint()
if args.format is not None:
    assert args.format in formats
    levels = args.format[:-3]
    print("GOT HERE!")
    print(args.other)
    if args.bench != "tensor3_elemadd" and args.bench != "tensor3_innerprod":
        assert args.bench is not None
        #$FROSTT_FORMATTED_TACO_PATH
        taco_format_orig_filename = "/nobackup/jadivara/sam/FROST_FORMATTED_TACO"
        outdir_other_name = os.path.join(outdir_name, args.name, args.bench)
        # outdir_other_name = os.path.join(outdir_name, args.name, 'other', otherfile[:-4])
        outdir_orig_path = Path(outdir_other_name)
        outdir_orig_path.mkdir(parents=True, exist_ok=True)

        name = None
        taco_format_orig_filename = os.path.join(taco_format_dirname, args.name + "_" + levels + '.txt')

        if args.bench == "tensor3_ttv":
            outdir_orig_name = os.path.join(outdir_name, args.name, args.bench, args.format)
            outdir_orig_path = Path(outdir_orig_name)
            outdir_orig_path.mkdir(parents=True, exist_ok=True)

            taco_format_orig_filename = "/nobackup/jadivara/sam/FROST_FORMATTED_TACO/" + args.name + "_" + levels + '.txt'
            parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format, hw_filename=args.hw)
            #Need this line? formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
            file_path_name = os.path.join(outdir_orig_name, "tensor_B_mode_shape")
            file1 = open(file_path_name, 'r')
            shape = [0]*3
            Lines = file1.readlines()
            count = 0
            # Strips the newline character
            for line in Lines:
                shape[count] = int(line)
                count += 1
            coo = inputCache.load(tensor, False)
            
            formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
            tensorname = 'c'
            vec = scipy.sparse.random(shape[1], 1, density=args.density, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            formatWriter.writeout_separate_vec(vec, dirname, tensorname)



            vec = scipy.sparse.random(shape[2], 1, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            formatWriter.writeout_separate_vec(vec, out_path, tensorname)
            #FormatWriter.writeout_separate_vec(vec, out_path, tensorname, tensorname)
            #formatWriter.writeout_separate_sparse_only()
        else:
            raise NotImplementedError

        assert name is not None, "Other tensor name was not set properly and is None"
        parse_taco_format(taco_format_orig_filename, outdir_other_name, name, args.format, hw_filename=args.hw)

    else:
        #this code is used for: tensor3_elemadd, tensor3_innerprod
        taco_format_orig_filename = os.path.join(taco_format_dirname, args.name + "_" + levels + '.txt')
        taco_format_shift_filename = os.path.join(taco_format_dirname, args.name + '_shift_' + levels + '.txt')

        # Original
        outdir_orig_name = os.path.join(outdir_name, args.name, args.bench, args.format)
        outdir_orig_path = Path(outdir_orig_name)
        outdir_orig_path.mkdir(parents=True, exist_ok=True)

        parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format, hw_filename=args.hw)

        # Shifted
        if args.shift:
            outdir_shift_name = os.path.join(outdir_name, args.name, args.bench, args.format)
            outdir_shift_path = Path(outdir_shift_name)
            outdir_shift_path.mkdir(parents=True, exist_ok=True)

            parse_taco_format(taco_format_shift_filename, outdir_shift_name, 'C', args.format, hw_filename=args.hw)
