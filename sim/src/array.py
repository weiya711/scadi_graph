from .base import Primitive


class Array(Primitive):
    def __init__(self, init_arr=None, size=1024, fill=0, **kwargs):
        super().__init__(**kwargs)

        self.size = size
        self.fill = fill
        if init_arr is None:
            self.arr = [self.fill] * self.size
        else:
            assert (isinstance(init_arr, list))
            self.arr = init_arr

        self.load_addrs = []
        self.store_vals = []
        self.load_en = False
        self.store_en = False

        self.curr_load = ''

    def update(self):
        if self.load_en and len(self.load_addrs) > 0:
            self.curr_load = self.load(self.load_addrs.pop(0))
            self.load_en = False

        if self.store_en and len(self.store_vals) > 0:
            store_tup = self.store_vals.pop(0)
            self.store(store_tup[0], store_tup[1])
            self.store_en = False

    def set_load(self, addr):
        if addr != '':
            self.load_en = True
            self.load_addrs.append(addr)

    def set_store(self, addr, vals):
        if addr != '' and vals != '':
            self.store_en = True
            self.store_vals.append((addr, vals))

    def get_arr(self):
        return self.arr

    def out_load(self):
        return self.curr_load

    def load(self, addr):
        # Special handling of loads of stop tokens
        if addr == 'S':
            return 'S'
        elif addr == 'D':
            self.done = True
            return 'D'
        elif addr > self.size:
            Exception("Address is out of array size bounds, please resize")
        else:

            return self.arr[addr]

    def store(self, addr, val):
        # Special handling of stores of stop tokens
        if addr == 'S' or val == 'S':
            return
        elif addr == 'D' or val == 'S':
            self.done = True
            return
        elif addr > self.size:
            Exception("Address is out of array size bounds, please resize")
        else:
            self.arr[addr] = val

    def reinit(self, init_arr):
        self.arr = init_arr

    def resize(self, size):
        if self.size > size:
            self.arr = self.arr[:size]
        else:
            self.arr = self.arr + [self.fill] * (size - self.size)
        self.size = size

    def clear(self, fill=0):
        self.arr = [fill for _ in range(self.size)]
