from .base import *


#################
# Read Scanners
#################


class CrdRdScan(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = 'S0'
        self.curr_crd = 'S0'

        self.in_ref = []

    def set_in_ref(self, in_ref):
        if in_ref != '':
            self.in_ref.append(in_ref)

    def out_ref(self):
        return self.curr_ref

    def out_crd(self):
        return self.curr_crd


# TODO: figure out how uncompressed read scans work with 'N' tokens
class UncompressCrdRdScan(CrdRdScan):
    def __init__(self, dim=0, **kwargs):
        super().__init__(**kwargs)

        self.start_addr = 0
        self.stop_addr = dim

        self.curr_in_ref = 0

        self.meta_dim = dim

    def update(self):

        # run out of coordinates, move to next input reference
        if self.curr_crd == '' or self.curr_crd == 'D':
            self.curr_crd = ''
            self.curr_ref = ''
        elif is_stkn(self.curr_crd):
            self.curr_in_ref = self.in_ref.pop(0)
            if self.curr_in_ref == 'D':
                self.curr_crd = 'D'
                self.curr_ref = 'D'
                self.done = True
                return
            else:
                self.curr_crd = 0
                self.curr_ref = self.curr_crd + (self.curr_in_ref * self.meta_dim)
        elif self.curr_crd >= self.meta_dim - 1:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.curr_crd = stkn
            self.curr_ref = stkn
        else:
            self.curr_crd += 1
            self.curr_ref = self.curr_crd + self.curr_in_ref * self.meta_dim

        if self.debug:
            print("DEBUG: U RD SCAN: \t "
                  "Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref)


class CompressedCrdRdScan(CrdRdScan):
    def __init__(self, crd_arr=[], seg_arr=[], **kwargs):
        super().__init__(**kwargs)

        self.crd_arr = crd_arr
        self.seg_arr = seg_arr

        self.start_addr = 0
        self.stop_addr = 0

        self.curr_addr = 0

        self.end_fiber = False
        self.curr_ref = None
        self.curr_crd = None
        self.emit_fiber_stkn = False

        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)

    def update(self):
        curr_in_ref = None
        # After Done token has been seen and outputted, do nothing
        if self.curr_crd == 'D' or self.curr_ref == 'D' or self.done:
            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
            self.curr_crd = ''
            self.curr_ref = ''

        # Scanner needs to emit stop token and the next element has finally arrived.
        # Previously set emit_fiber_stkn to True but wait on next in_ref
        elif len(self.in_ref) > 0 and self.emit_fiber_stkn:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.curr_crd = stkn
            self.curr_ref = stkn

            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
            self.emit_fiber_stkn = False

        # There exists another input reference at the segment and
        # either at the start of computation or end of fiber
        elif len(self.in_ref) > 0 and (self.end_fiber or (self.curr_crd is None or self.curr_ref is None)):
            if self.curr_crd is None or self.curr_ref is None:
                assert (self.curr_crd == self.curr_ref)
            self.end_fiber = False

            curr_in_ref = self.in_ref.pop(0)

            # Input reference is out of bounds
            if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_slen:
                raise Exception('Not enough elements in seg array')

            # Input reference is a stop or done token, so forward that token (and set done if done token)
            elif is_stkn(curr_in_ref) or curr_in_ref == 'D':
                self.curr_addr = 0
                self.stop_addr = 0
                self.start_addr = 0
                self.curr_crd = curr_in_ref
                self.curr_ref = curr_in_ref
                self.end_fiber = True
                if curr_in_ref == 'D':
                    self.done = True

            # See 'N' 0-token which immediately emits a stop token and ends the fiber
            elif is_0tkn(curr_in_ref):
                self.end_fiber = True

                if len(self.in_ref) > 0:
                    next_in = self.in_ref[0]
                    if is_stkn(next_in):
                        self.in_ref.pop(0)
                        stkn = increment_stkn(next_in)
                    else:
                        stkn = 'S0'
                else:
                    self.emit_fiber_stkn = True
                    stkn = ''
                self.curr_crd = stkn
                self.curr_ref = stkn
                self.curr_addr = 0
                self.stop_addr = 0
                self.start_addr = 0

            # Default case where input reference is an integer value
            # which means to get the segment
            else:
                self.start_addr = self.seg_arr[curr_in_ref]
                self.stop_addr = self.seg_arr[curr_in_ref + 1]
                self.curr_addr = self.start_addr

                # This case is if the segment has no coordinates (i.e. 5, 5)
                if self.curr_addr >= self.stop_addr:
                    # End of fiber, get next input reference
                    self.end_fiber = True

                    if len(self.in_ref) > 0:
                        next_in = self.in_ref[0]
                        if is_stkn(next_in):
                            self.in_ref.pop(0)
                            stkn = increment_stkn(next_in)
                        else:
                            stkn = 'S0'
                    else:
                        self.emit_fiber_stkn = True
                        stkn = ''
                    self.curr_crd = stkn
                    self.curr_ref = stkn

                # Default behave normally and emit the coordinates in the segment
                else:
                    self.curr_crd = self.crd_arr[self.curr_addr]
                    self.curr_ref = self.curr_addr

        # Finished emitting coordinates and have reached the end of the fiber for this level
        elif (self.curr_addr == self.stop_addr - 1 or self.curr_addr == self.meta_clen - 1) and \
                self.curr_crd is not None and self.curr_ref is not None:
            # End of fiber, get next input reference
            self.end_fiber = True

            if len(self.in_ref) > 0:
                next_in = self.in_ref[0]
                if is_stkn(next_in):
                    self.in_ref.pop(0)
                    stkn = increment_stkn(next_in)
                else:
                    stkn = 'S0'
            else:
                self.emit_fiber_stkn = True
                stkn = ''
            self.curr_crd = stkn
            self.curr_ref = stkn
            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0

        # Base case: increment address and reference by 1 and get next coordinate
        elif len(self.in_ref) > 0 and self.curr_crd is not None and self.curr_ref is not None:
            self.curr_addr += 1
            self.curr_ref = self.curr_addr
            self.curr_crd = self.crd_arr[self.curr_addr]

        # Default stall (when done)
        elif self.curr_crd is not None and self.curr_ref is not None:
            self.curr_ref = ''
            self.curr_crd = ''

        if self.debug:
            print("DEBUG: C RD SCAN: \t "
                  "Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref, "\t curr addr:", self.curr_addr,
                  "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                  "\t end fiber:", self.end_fiber, "\t curr input:", curr_in_ref)


# ---------------- BV --------------#

class BVRdScanSuper(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = 'S0'
        self.curr_bv = 'S0'

        self.in_ref = []

    def set_in_ref(self, in_ref):
        if in_ref != '':
            self.in_ref.append(in_ref)

    def out_ref(self):
        return self.curr_ref

    def out_bv(self):
        return self.curr_bv


class BVRdScan(BVRdScanSuper):
    def __init__(self, bv_arr=None, dim=4, nbits=4, **kwargs):
        super().__init__(**kwargs)

        self.bv_arr = bv_arr if bv_arr is not None else [2 ** nbits - 1] * dim

        self.curr_addr = 0

        self.end_fiber = False
        self.curr_ref = None
        self.curr_bv = None
        self.emit_fiber_stkn = False

        self.meta_blen = len(bv_arr)
        self.meta_nbits = nbits
        self.meta_dim = dim

    def _get_bv_ref(self, addr):
        assert isinstance(addr, int), "Addresses must be integers"
        if addr <= 0:
            return 0
        bits = sum(map(popcount, self.bv_arr[:addr]))
        return bits

    def update(self):
        curr_in_ref = None
        if self.curr_bv == 'D' or self.curr_ref == 'D' or self.done:
            self.curr_addr = 0
            self.curr_bv = ''
            self.curr_ref = ''
        elif len(self.in_ref) > 0 and self.emit_fiber_stkn:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.curr_bv = stkn
            self.curr_ref = stkn

            self.curr_addr = 0
            self.emit_fiber_stkn = False
        # There exists another input reference at the segment and
        # either at the start of computation or end of fiber
        elif len(self.in_ref) > 0 and (self.end_fiber or (self.curr_bv is None or self.curr_ref is None)):
            if self.curr_bv is None or self.curr_ref is None:
                assert (self.curr_bv == self.curr_ref)
            self.end_fiber = False

            curr_in_ref = self.in_ref.pop(0)
            if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_blen:
                raise Exception('Not enough elements in seg array')

            # See 'N' 0-token which immediately emits a stop token and ends the fiber
            # TODO: need to figure out how this will work with bitvectors
            elif is_0tkn(curr_in_ref):
                self.end_fiber = True

                if len(self.in_ref) > 0:
                    next_in = self.in_ref[0]
                    if is_stkn(next_in):
                        self.in_ref.pop(0)
                        stkn = increment_stkn(next_in)
                    else:
                        stkn = 'S0'
                else:
                    self.emit_fiber_stkn = True
                    stkn = ''
                self.curr_bv = stkn
                self.curr_ref = stkn
                self.curr_addr = 0

            elif is_stkn(curr_in_ref) or curr_in_ref == 'D':
                self.curr_addr = 0
                self.curr_bv = curr_in_ref
                self.curr_ref = curr_in_ref
                self.end_fiber = True
                if curr_in_ref == 'D':
                    self.done = True
            else:
                self.curr_addr = curr_in_ref
                # End of fiber, get next input reference
                self.end_fiber = True

                self.emit_fiber_stkn = True
                self.curr_bv = self.bv_arr[self.curr_addr]

                self.curr_ref = self._get_bv_ref(self.curr_addr)
        elif self.curr_bv is not None and self.curr_ref is not None:
            # Default stall (when done)
            self.curr_ref = ''
            self.curr_bv = ''

        if self.debug:
            print("DEBUG: C RD SCAN: \t "
                  "Curr bv:", self.curr_bv, "\t curr ref:", self.curr_ref, "\t curr addr:", self.curr_addr,
                  "\t end fiber:", self.end_fiber, "\t curr input:", curr_in_ref)
