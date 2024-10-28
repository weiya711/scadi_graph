from sam.onyx.hw_nodes.hw_node import *


class PassThroughNode(HWNode):
    def __init__(self, name=None, conn_to_tensor=None) -> None:
        super().__init__(name=name)

        self.conn_to_tensor = conn_to_tensor
        self.tensor_to_conn = {}
        if conn_to_tensor is not None:
            for conn, tensor in self.conn_to_tensor.items():
                self.tensor_to_conn[tensor] = conn

    def connect(self, other, edge, kwargs=None):

        from sam.onyx.hw_nodes.broadcast_node import BroadcastNode
        from sam.onyx.hw_nodes.compute_node import ComputeNode
        from sam.onyx.hw_nodes.glb_node import GLBNode
        from sam.onyx.hw_nodes.buffet_node import BuffetNode
        from sam.onyx.hw_nodes.memory_node import MemoryNode
        from sam.onyx.hw_nodes.read_scanner_node import ReadScannerNode
        from sam.onyx.hw_nodes.write_scanner_node import WriteScannerNode
        from sam.onyx.hw_nodes.intersect_node import IntersectNode
        from sam.onyx.hw_nodes.reduce_node import ReduceNode
        from sam.onyx.hw_nodes.lookup_node import LookupNode
        from sam.onyx.hw_nodes.merge_node import MergeNode
        from sam.onyx.hw_nodes.repeat_node import RepeatNode
        from sam.onyx.hw_nodes.repsiggen_node import RepSigGenNode
        from sam.onyx.hw_nodes.crdhold_node import CrdHoldNode
        from sam.onyx.hw_nodes.fiberaccess_node import FiberAccessNode

        new_conns = None
        pass_through = self.get_name()

        other_type = type(other)
        # print(other_type)

        if other_type == WriteScannerNode:
            wr_scan = other.get_name()
            new_conns = {
                'pass_through_to_wr_scan': [
                    ([(pass_through, "stream_out"), (wr_scan, "block_wr_in")], 17),
                ]
            }
            return new_conns
        elif other_type == ReadScannerNode:
            rd_scan = other.get_name()
            new_conns = {
                'pass_through_to_rd_scan': [
                    ([(pass_through, "stream_out"), (rd_scan, f"us_pos_in")], 17),
                ]
            }
            return new_conns
        elif other_type == RepeatNode:
            repeat = other.get_name()
            new_conns = {
                'pass_through_to_repeat': [
                    # send output to rd scanner
                    ([(pass_through, "stream_out"), (repeat, "proc_data_in")], 17),
                ]
            }
            return new_conns
        elif other_type == IntersectNode:
            comment = edge.get_attributes()['comment'].strip('"')
            try:
                tensor = comment.split("-")[1]
            except Exception:
                try:
                    tensor = comment.split("_")[1]
                except Exception:
                    tensor = comment

            other_isect = other.get_name()
            isect_conn = self.get_connection_from_tensor(tensor)
            other_isect_conn = other.get_connection_from_tensor(tensor)

            edge_type = edge.get_attributes()['type'].strip('"')

            if 'crd' in edge_type:
                new_conns = {
                    f'pass_through_to_isect': [
                        ([(pass_through, "stream_out"), (other_isect, f"coord_in_{other_isect_conn}")], 17),
                    ]
                }
            elif 'ref' in edge_type:
                new_conns = {
                    f'pass_through_to_isect': [
                        ([(pass_through, "stream_out"), (other_isect, f"pos_in_{other_isect_conn}")], 17),
                    ]
                }
            return new_conns

        elif other_type == MergeNode:
            edge_attr = edge.get_attributes()
            crddrop = other.get_name()
            crd_drop_outer = other.get_outer()
            comment = edge_attr['comment'].strip('"')
            conn = 0
            # okay this is dumb, stopgap until we can have super consistent output
            try:
                mapped_to_conn = comment.split("-")[1]
            except Exception:
                try:
                    mapped_to_conn = comment.split("_")[1]
                except Exception:
                    mapped_to_conn = comment
            if crd_drop_outer in mapped_to_conn:
                conn = 1

            if 'use_alt_out_port' in edge_attr:
                out_conn = 'block_rd_out'
            elif ('vector_reduce_mode' in edge_attr):
                if (edge_attr['vector_reduce_mode']):
                    out_conn = 'pos_out'
            else:
                out_conn = 'coord_out'

            new_conns = {
                f'rd_scan_to_crddrop_{conn}': [
                    ([(pass_through, "stream_out"), (crddrop, f"coord_in_{conn}")], 17),
                ]
            }

            return new_conns
        elif other_type == RepSigGenNode:
            rsg = other.get_name()
            new_conns = {
                f'pass_through_to_rsg': [
                    ([(pass_through, "stream_out"), (rsg, f"base_data_in")], 17),
                ]
            }
        elif other_type == FiberAccessNode:
            # fa = other.get_name()
            assert kwargs is not None
            assert 'flavor_that' in kwargs
            that_flavor = other.get_flavor(kwargs['flavor_that'])
            init_conns = self.connect(that_flavor, edge)
            final_conns = other.remap_conns(init_conns, kwargs['flavor_that'])
            return final_conns

        else:
            raise NotImplementedError(f'Cannot connect Pass Through Node to {other_type}')

        return new_conns

    def get_connection_from_tensor(self, tensor):
        # print(self.tensor_to_conn)
        return self.tensor_to_conn[tensor]

    def update_input_connections(self):
        self.num_inputs_connected += 1

    def get_num_inputs(self):
        return self.num_inputs_connected

    def configure(self, attributes):
        print("PASSTHROUGH Configure", attributes)

        placeholder = 1
        cfg_kwargs = {
            'placeholder': placeholder
        }
        return (placeholder), cfg_kwargs
