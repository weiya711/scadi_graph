from ast import dump
from operator import mod
from venv import create
import numpy
import random
import scipy.sparse as ss
import tempfile

import torch

from sam.onyx.fiber_tree import *
import argparse
import math
import csv
import os
from sam.sim.test.test import *


class MatrixGenerator:

    def __init__(self, name='B', shape=None, mode_ordering=None, block_size=1, block_naive=True, sparsity=0.6,
                 format='CSF', dump_dir=None, tensor=None, value_cap=None, transpose=False) -> None:

        # assert dimension is not None
        # self.dimension = dimension
        self.shape = shape
        self.array = None
        self.blocked_array = None
        self.sparsity = sparsity
        self.format = format
        self.name = name
        self.mode_ordering = mode_ordering
        self.block_size = block_size
        self.block_naive = block_naive
        self.transpose = transpose
        if value_cap is None:
            self.value_cap = int(math.pow(2, 8)) - 1
        else:
            self.value_cap = value_cap

        self.fiber_tree = None
        self.tmp_fiber_tree = None

        if dump_dir is not None:
            self.dump_dir = dump_dir
            if not os.path.isdir(self.dump_dir):
                os.mkdir(self.dump_dir)
            # else:
            #     # Otherwise clean it
            #     for filename in os.listdir(self.dump_dir):
            #         ret = os.remove(self.dump_dir + "/" + filename)
        else:
            self.dump_dir = tempfile.gettempdir()

        if tensor is not None:
            self.array = tensor
            self.shape = self.array.shape
            if mode_ordering is None:
                self.mode_ordering = range(len(shape))
        else:
            assert shape is not None
            # assert mode_ordering is not None
            if mode_ordering is None:
                self.mode_ordering = range(len(shape))
            assert len(shape) == len(mode_ordering)
            if mode_ordering != list(range(len(shape))):
                self.shape = [shape[i] for i in mode_ordering]
            self._create_matrix(value_cap=self.value_cap)
        self._create_fiber_tree()

    def _create_matrix(self, value_cap=int(math.pow(2, 8)) - 1):
        '''
        Routine to create the actual matrix from the dimension/shape
        '''
        self.array = numpy.random.randint(low=0, high=value_cap, size=self.shape)
        
        for idx, x in numpy.ndenumerate(self.array):
            rand = random.random()
            if self.block_size > 1:
                if (idx[-2] % self.block_size == 0) and (idx[-1]) % self.block_size == 0:
                    if rand < self.sparsity:
                        self.array[..., idx[-2]:idx[-2]+self.block_size, idx[-1]:idx[-1]+self.block_size] = 0
            else:
                if rand < self.sparsity:
                    self.array[idx] = 0

    def _create_fiber_tree(self):
        # self.fiber_tree = FiberTree(tensor=self.array if self.block_naive else self.array[...,
                                    # self.shape[-2]:self.block_size,:self.shape[-1]:self.block_size])
        self.fiber_tree = FiberTree(tensor=self.array)
        if len(self.shape) > 1:
            self.tmp_fiber_tree = FiberTree(tensor=self.array[..., :self.shape[-2]:self.block_size,
                                            :self.shape[-1]:self.block_size])
        else:
            self.tmp_fiber_tree = FiberTree(tensor=self.array)

    def transpose_tensor(self, axes=None):
        '''
        Transposes the tensor using specified axes
        '''
        if axes is not None:
            self.array = numpy.transpose(self.array, axes)
            self.shape = self.array.shape
            self.fiber_tree = FiberTree(tensor=self.array)

    def set_dump_dir(self, dump_dir):
        if dump_dir is not None:
            self.dump_dir = dump_dir
            if not os.path.isdir(self.dump_dir):
                os.mkdir(self.dump_dir)
            # else:
            #     # Otherwise clean it
            #     for filename in os.listdir(self.dump_dir):
            #         ret = os.remove(self.dump_dir + "/" + filename)

    def dump_outputs(self, format=None, tpose=False, dump_shape=True, glb_override=False, glb_dump_dir=None):
        '''
        Dump the matrix into many files depending on matrix format
        '''

        use_dir = self.dump_dir
        print_hex = False
        if glb_override:
            use_dir = glb_dump_dir
            print_hex = True

        print(f"Using dump directory - {use_dir}")

        # all_zeros = not np.any(self.array, out=self.array)
        all_zeros = False

        debug_coord_arr = []
        debug_seg_arr = []
        debug_val_arr = []
        # Transpose it first if necessary
        # if tpose is True:
        #     self.array = numpy.transpose(self.array)
        #     self.shape = self.array.shape
        #     self.fiber_tree = FiberTree(tensor=self.array)

        if format is not None:
            self.format = format

        if self.format == 'CSF':
            # Handle the all zeros case...
            if all_zeros:
                fake_lines_seg = ["0000",
                                  "0000"]
                fake_lines_crd = ["0000"]
                # If it's a scalar/length 1 vec
                if len(self.shape) == 1 and self.shape[0] == 1:
                    fake_lines_val = ["0000"]
                else:
                    fake_lines_val = ["0000"]
                # for mode in range(len(self.array.shape)):
                for mode in self.mode_ordering:
                    if glb_override:
                        lines = [len(fake_lines_seg), *fake_lines_seg, len(fake_lines_crd), *fake_lines_crd]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_{mode}", dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(fake_lines_seg, name=f"tensor_{self.name}_mode_{mode}_seg",
                                         dump_dir=use_dir, hex=print_hex)
                        self.write_array(fake_lines_crd, name=f"tensor_{self.name}_mode_{mode}_crd",
                                         dump_dir=use_dir, hex=print_hex)
                if glb_override:
                    lines = [len(fake_lines_val), *fake_lines_val]
                    self.write_array(fake_lines_val, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir,
                                     hex=print_hex)
                else:
                    self.write_array(fake_lines_val, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir,
                                     hex=print_hex)

                return

            # In CSF format, need to iteratively create seg/coord arrays
            tmp_lvl_list = []
            small_tmp_lvl_list = []
            tmp_lvl_list.append(self.fiber_tree.get_root())
            small_tmp_lvl_list.append(self.tmp_fiber_tree.get_root())

            seg_arr, coord_arr = None, None
            if self.block_size > 1:
                seg_arr, coord_arr = self._dump_csf(small_tmp_lvl_list)
            else:
                seg_arr, coord_arr = self._dump_csf(tmp_lvl_list)
            if glb_override:
                lines = [len(seg_arr), *seg_arr, len(coord_arr), *coord_arr]
                self.write_array(lines, name=f"tensor_{self.name}_mode_{self.mode_ordering[0]}", dump_dir=use_dir,
                                 hex=print_hex)
            else:
                self.write_array(seg_arr, name=f"tensor_{self.name}_mode_{self.mode_ordering[0]}_seg", dump_dir=use_dir,
                                 hex=print_hex)
                self.write_array(coord_arr, name=f"tensor_{self.name}_mode_{self.mode_ordering[0]}_crd",
                                 dump_dir=use_dir, hex=print_hex)

            at_vals = False
            # TODO: Might need a try catch
            i = 1

            tmp_lst = tmp_lvl_list if self.block_size == 1 else small_tmp_lvl_list
            while at_vals is False:
                # Make the next level of fibers - basically BFS but segmented across depth of tree
                next_tmp_lvl_list = []
                for fib in tmp_lst:
                    crd_payloads_tmp = fib.get_coord_payloads()
                    if type(crd_payloads_tmp[0][1]) is not FiberTreeFiber:
                        at_vals = True
                        for crd, pld in crd_payloads_tmp:
                            next_tmp_lvl_list.append(pld)
                    else:
                        for crd, pld in crd_payloads_tmp:
                            next_tmp_lvl_list.append(pld)
                tmp_lst = next_tmp_lvl_list
                if at_vals:
                    # If at vals, we don't need to dump csf, we have the level
                    if glb_override:
                        lines = [len(tmp_lst), *tmp_lst]
                        # self.write_array(tmp_lvl_list, name=f"tensor_{self.name}_mode_vals" dump_dir=use_dir)
                        self.write_array(lines, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                    else:
                        # reached_full_vals = False
                        # if self.block_size > 1:
                        #     tmp_lst = tmp_lvl_list
                        #     # Retrieve values from full tensor 
                        #     while not reached_full_vals:
                        #         next_tmp_lvl_list = []
                        #         for fib in tmp_lst:
                        #             crd_payloads_tmp = fib.get_coord_payloads()
                        #             if type(crd_payloads_tmp[0][1]) is not FiberTreeFiber:
                        #                 reached_full_vals = True
                        #                 for crd, pld in crd_payloads_tmp:
                        #                     next_tmp_lvl_list.append(pld)
                        #             else:
                        #                 for crd, pld in crd_payloads_tmp:
                        #                     next_tmp_lvl_list.append(pld)
                        #         tmp_lst = next_tmp_lvl_list
                        if self.block_size == 1:
                            self.write_array(tmp_lst, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir,
                                            hex=print_hex)
                        else:
                            self.write_blocked_array(self.array, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, skip_zeros=True)
                else:
                    seg_arr, coord_arr = self._dump_csf(tmp_lst)
                    if glb_override:
                        lines = [len(seg_arr), *seg_arr, len(coord_arr), *coord_arr]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_{self.mode_ordering[i]}",
                                         dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(seg_arr, name=f"tensor_{self.name}_mode_{self.mode_ordering[i]}_seg",
                                         dump_dir=use_dir, hex=print_hex)
                        self.write_array(coord_arr, name=f"tensor_{self.name}_mode_{self.mode_ordering[i]}_crd",
                                         dump_dir=use_dir, hex=print_hex)
                i = i + 1
        elif self.format == "UNC":
            flat_array = []
            for val in numpy.nditer(self.array):
                flat_array.append(val)
            if glb_override:
                lines = [len(flat_array), *flat_array]
                self.write_array(lines, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
            else:
                if self.block_size == 1:
                    self.write_array(flat_array, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                else:
                    self.write_blocked_array(self.array, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir)
        elif self.format == "COO":
            crd_dict = dict()
            order = len(self.array.shape)
            for i in range(order + 1):
                crd_dict[i] = []
            it = np.nditer(self.array, flags=['multi_index'])
            while not it.finished:
                crd_dict[order].append(it[0])
                point = it.multi_index
                for i in range(order):
                    crd_dict[i].append(point[i])
                is_not_finished = it.iternext()
            for key in crd_dict:
                if key == order:
                    if glb_override:
                        lines = [len(crd_dict[key]), *crd_dict[key]]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir, hex=print_hex)
                    else:
                        self.write_array(crd_dict[key], name=f"tensor_{self.name}_mode_vals", dump_dir=use_dir,
                                         hex=print_hex)
                else:
                    if glb_override:
                        lines = [len(crd_dict[key]), *crd_dict[key]]
                        self.write_array(lines, name=f"tensor_{self.name}_mode_{key}_crd", dump_dir=use_dir,
                                         hex=print_hex)
                    else:
                        self.write_array(crd_dict[key],
                                         name=f"tensor_{self.name}_mode_{key}_crd",
                                         dump_dir=use_dir,
                                         hex=print_hex)

        if dump_shape:
            # final_shape = self.array.shape
            final_shape = [x if i < len(self.array.shape) - 2 else x // self.block_size for i, x in enumerate(self.array.shape)]
            # final_shape = final_shape[self.mode_ordering]
            shape = []
            for elem in self.mode_ordering:
                shape.append(final_shape[elem])

            self.write_array(shape, name=f"tensor_{self.name}_mode_shape", dump_dir=use_dir, hex=print_hex)

        # Transpose it back
        if tpose is True:
            self.array = numpy.transpose(self.array)
            self.shape = self.array.shape
            self.fiber_tree = FiberTree(tensor=self.array)

    def _dump_csf(self, level_list):
        """ Dumps the csf-based seg/coord array for each level, unless it is a vals list

        Args:
            level_list (list): list of fibers at this level
        """

        seg_arr = []
        seg_running = 0
        coord_arr = []
        seg_arr.append(0)
        for level_fiber in level_list:
            # For each fiber in the level, the seg is just adding the running length, coord array is simple
            lf_crd_pld = level_fiber.get_coord_payloads()
            seg_running += len(lf_crd_pld)
            seg_arr.append(seg_running)
            for crd, pld in lf_crd_pld:
                coord_arr.append(crd)

        return seg_arr, coord_arr

    def write_array(self, str_list, name, dump_dir=None, hex=False, integer=False, num_repeats=1):
        """Write an array/list to a file

        Args:
            list (list): array/list of values
            name (str): name of file
        """
        if dump_dir is None:
            dump_dir = self.dump_dir

        full_path = dump_dir + "/" + name
        with open(full_path, "w+") as wr_file:
            for item in str_list:
                if integer:
                    item_int = int(item)
                else:
                    item_int = item
                if hex:
                    wr_file.write(f"{item_int:04X}\n")
                else:
                    for _ in range(num_repeats):
                        wr_file.write(f"{item_int}\n")

    def write_blocked_array(self, str_list, name, dump_dir=None, integer=False, hex=False, skip_zeros=False):
        if dump_dir is None:
            dump_dir = self.dump_dir

        tiles = tile_and_unroll_nd(str_list, [self.block_size, self.block_size])

        if skip_zeros:
            print("Skipping zeros")
            tiles = [tile for tile in tiles if numpy.sum(tile) != 0]

        tiles = np.concatenate(tiles)

        full_path = dump_dir + "/" + name
        with open(full_path, "w+") as wr_file:
            for item in tiles:
                if integer:
                    item_int = int(item)
                else:
                    item_int = item
                if hex:
                    wr_file.write(f"{item_int:04X}\n")
                else:
                    wr_file.write(f"{item_int}\n")

    def get_shape(self):
        return self.shape

    def get_name(self):
        return self.name

    def get_matrix(self):
        return self.array

    def __str__(self):
        return str(self.array)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, val):
        self.array[key] = val


def tile_and_unroll_nd(array, tile_size):
    """
    Generalized function to tile and unroll a multidimensional array.

    Parameters:
    - array: Input numpy array of any dimensionality.
    - tile_size: A tuple that specifies the size of the tiles for the last dimensions.

    Returns:
    - A list of flattened tiles.
    """
    array_shape = array.shape
    num_non_tiled_dims = len(array_shape) - len(tile_size)  # Non-tiled dimensions
    tiled_dims = array_shape[num_non_tiled_dims:]  # Dimensions that will be tiled

    # Calculate the number of tiles along the tiling dimensions
    tiled_sizes = [dim // tile for dim, tile in zip(tiled_dims, tile_size)]

    # Prepare a list to collect the tiles
    tiles = []

    # Iterate over the non-tiled dimensions
    for index in np.ndindex(*array_shape[:num_non_tiled_dims]):
        # Iterate over the tiled dimensions and extract each tile
        for tile_index in np.ndindex(*tiled_sizes):
            # Slice the array to extract the current tile
            slices = tuple(
                slice(idx * size, (idx + 1) * size) for idx, size in zip(tile_index, tile_size)
            )
            full_slices = index + slices  # Combine non-tiled and tiled indices
            tile = array[full_slices]
            if type(tile) == torch.Tensor:
                tile = tile.numpy()
            # Flatten the tile and append to the list
            tiles.append(tile.flatten())

    return tiles


def get_runs(v1, v2):
    """Get the average run length/runs of each vector

    Args:
        v1 (vector): vector 1
        v2 (vector): vector 2
    """

    # Ensure both are vectors of the same length
    assert len(v1.shape) == 1
    assert len(v2.shape) == 1
    assert v1.shape[0] == v2.shape[0]

    run_idx = [[], []]

    # First coiterate to get to first mismatch
    on_run = False
    run_start = None
    run_end = None
    run_side = None

    last_nonzero_v1 = None
    last_nonzero_v2 = None

    for idx, val in numpy.ndenumerate(v1):
        v2_val = v2[idx]
        idx = idx[0]
        if (val == 0 and v2_val != 0) or (val != 0 and v2_val == 0):
            # If run side is 0 and val is 0, we continue,
            if not on_run:
                # If not on run, determine the side and run_start, on_run
                if val != 0:
                    run_side = 0
                else:
                    run_side = 1
                run_start = idx
                on_run = True
            elif val != 0 and run_side == 0:
                # Continuing the run
                pass
            elif val != 0 and run_side == 1:
                # Here we are seeing the run end on the other side
                run_end = last_nonzero_v2
                run_idx[run_side].append((run_start, run_end))
                # Now we are starting a new run
                run_side = 0
                run_start = idx
            elif v2_val != 0 and run_side == 1:
                # Continuing the run
                pass
            elif v2_val != 0 and run_side == 0:
                # Here we are seeing the run end on the other side
                run_end = last_nonzero_v1
                run_idx[run_side].append((run_start, run_end))
                # Now we are starting a new run
                run_side = 1
                run_start = idx
        elif val == 0 and v2_val == 0:
            # If both are 0 it's fine
            pass
        else:
            # If both are 1 the run is over at the previous nonzero value, and we are no longer on a run (assuming we were)
            if on_run:
                if run_side == 0:
                    run_end = last_nonzero_v1
                else:
                    run_end = last_nonzero_v2
                run_idx[run_side].append((run_start, run_end))
                on_run = False

        if val != 0:
            last_nonzero_v1 = idx
        if v2_val != 0:
            last_nonzero_v2 = idx

    # Now off the end, if we have started a run on either side, we should terminate it
    if on_run:
        run_idx[run_side].append((run_start, v1.shape[0] - 1))

    return run_idx


def get_run_lengths(run_idx, v1, v2):
    run_len = [[], []]

    for side in range(2):
        tmp_vec = v1
        if side == 1:
            tmp_vec = v2
        for idx_tuple in run_idx[side]:
            tmp_run_len = 0
            idx1, idx2 = idx_tuple
            for idx in range(idx2 + 1 - idx1):
                if tmp_vec[idx + idx1] != 0:
                    tmp_run_len += 1
            run_len[side].append(tmp_run_len)

    return run_len


def delete_run(vec, run_list):
    """Delete random run from a vector

    Args:
        vec (_type_): _description_
        run_list (_type_): _description_
    """

    # Early out
    if len(run_list) == 0:
        return

    random_idx = random.randint(0, vec.size)
    run_to_del = mod(random_idx, len(run_list))

    idx1, idx2 = run_list[run_to_del]
    for i in range(idx2 + 1 - idx1):
        vec[i + idx1] = 0


def run_statistics(name, seed, shape, dump_dir, sparsity):
    random.seed(seed)
    numpy.random.seed(seed)
    vec1 = MatrixGenerator(name=name, shape=shape, dump_dir=dump_dir, sparsity=sparsity)
    vec2 = MatrixGenerator(name=f"{name}_alt", shape=shape, dump_dir=dump_dir, sparsity=sparsity)

    # Now delete runs from the first
    for i in range(5):
        run_idx = get_runs(vec1.get_matrix(), vec2.get_matrix())
        delete_run(vec1.get_matrix(), run_idx[0])

    run_list = get_run_lengths(run_idx, vec1.get_matrix(), vec2.get_matrix())

    if len(run_list[0]) > 0:
        avg1 = sum(run_list[0]) / len(run_list[0])
    else:
        avg1 = 0
    if len(run_list[1]) > 0:
        avg2 = sum(run_list[1]) / len(run_list[1])
    else:
        avg2 = 0

    return (avg1, avg2)


def create_matrix_from_point_list(name, pt_list, shape) -> MatrixGenerator:
    mat_base = numpy.zeros(shape)
    dims = len(shape)
    for pt_idx in range(len(pt_list[0])):
        pt_base = []
        for i in range(dims):
            pt_base.append(pt_list[i][pt_idx])
        mat_base[tuple(pt_base)] = pt_list[dims][pt_idx]

    mg = MatrixGenerator(name=f"{name}", shape=shape, sparsity=0.7, format='CSF', dump_dir=None, tensor=mat_base)
    return mg


def convert_aha_glb_output_file(glbfile, output_dir):
    glbfile_s = os.path.basename(glbfile).rstrip(".txt")

    if 'mode_vals' in glbfile:
        # num_blocks = 1
        files = [f"{output_dir}/{glbfile_s}"]
    else:
        # num_blocks = 2
        files = [f"{output_dir}/{glbfile_s}_seg",
                 f"{output_dir}/{glbfile_s}_crd"]

    straightline = []

    # Straighten the file out
    with open(glbfile, "r") as glbfile_h:
        file_contents = glbfile_h.readlines()
        for line in file_contents:
            sp_line = line.strip().split(" ")
            for sp_line_tok in sp_line:
                sp_line_tok_stripped = sp_line_tok.strip()
                straightline.append(int(sp_line_tok_stripped, base=16))

    # Now we have straightline having the items in order
    # Now write them to the output
    sl_ptr = 0
    for file_path in files:
        num_items = straightline[sl_ptr]
        sl_ptr += 1
        with open(file_path, "w+") as fh_:
            for _ in range(num_items):
                fh_.write(f"{straightline[sl_ptr]:04X}\n")
                sl_ptr += 1


def get_tensor_from_files(name, files_dir, shape, base=10, format='CSF', early_terminate=None,
                          mode_ordering=None) -> MatrixGenerator:
    all_files = os.listdir(files_dir)
    dims = len(shape)

    # Get vals first since all formats will have vals
    val_f = [fil for fil in all_files if name in fil and f'vals' in fil][0]
    vals = read_inputs(f"{files_dir}/{val_f}", intype=int, base=base, early_terminate=early_terminate)

    mg = None
    if dims == 1 and shape[0] == 1:  # scalar
        mat_sc = numpy.zeros([1])
        mat_sc[0] = vals[0]
        mg = MatrixGenerator(name=name, shape=shape, tensor=mat_sc)
    elif format == 'CSF':
        created_empty = False
        segs = []
        crds = []
        modes = []
        if mode_ordering is not None:
            modes = mode_ordering
        else:
            modes = range(dims)
        for mode in modes:
            seg_f = [fil for fil in all_files if name in fil and f'mode_{mode}' in fil and 'seg' in fil][0]
            crd_f = [fil for fil in all_files if name in fil and f'mode_{mode}' in fil and 'crd' in fil][0]
            seg_t_ = read_inputs(f"{files_dir}/{seg_f}", intype=int, base=base, early_terminate=early_terminate)
            segs.append(seg_t_)
            # Empty matrix...
            if mode == 0 and len(seg_t_) == 2 and seg_t_[0] == 0 and seg_t_[1] == 0:
                mg = MatrixGenerator(name=name, shape=shape, sparsity=1.0)
                created_empty = True
                break
            crd_t_ = read_inputs(f"{files_dir}/{crd_f}", intype=int, base=base, early_terminate=early_terminate)
            crds.append(crd_t_)
        if not created_empty:
            pt_list = get_point_list(crds, segs, val_arr=vals)
            mg = create_matrix_from_point_list(name, pt_list, shape)
    elif format == 'COO':
        crds = []
        for mode in range(dims):
            crd_f = [fil for fil in all_files if name in fil and f'{mode}' in fil and 'crd' in fil][0]
            crds.append(read_inputs(f"{files_dir}/{crd_f}", intype=int, base=base, early_terminate=early_terminate))

        pt_list = copy.deepcopy(crds)
        pt_list.append(vals)
        mg = create_matrix_from_point_list(name, pt_list, shape)

    return mg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--dump_dir", type=str, default=None)
    parser.add_argument("--name", type=str, default='B')
    parser.add_argument("--shape", type=int, nargs="*", default=[10])
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--output_csv", type=str, default="runs.csv")
    args = parser.parse_args()

    seed = args.seed
    sparsity = args.sparsity
    name = args.name
    dump_dir = args.dump_dir
    shape = args.shape
    csv_out = args.output_csv

    averages_list = []
    for override_seed in range(1000):
        avg1, avg2 = run_statistics(name, override_seed, shape, dump_dir, sparsity)
        averages_list.append([override_seed, avg1, avg2])

    # Write a csv out
    fields = ['seed', 'average_runs_1', 'average_runs_2']

    with open(csv_out, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(averages_list)

    quit()

    random.seed(seed)
    numpy.random.seed(seed)
    vec1 = MatrixGenerator(name=name, shape=shape, dump_dir=dump_dir, sparsity=sparsity)
    vec2 = MatrixGenerator(name=f"{name}_alt", shape=shape, dump_dir=dump_dir, sparsity=sparsity)

    run_list = get_runs(vec1.get_matrix(), vec2.get_matrix())
    # mg.dump_outputs()

    avg1 = sum(run_list[0]) / len(run_list[0])
    avg2 = sum(run_list[1]) / len(run_list[1])

    print(f"Average Run Length V1: {avg1}")
    print(f"Average Run Length V2: {avg2}")
