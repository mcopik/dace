# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

# General DaCe imports
import dace
from dace import data as dt
from dace.sdfg import nodes

# Code generator imports and helpers
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.cpp import cpp_array_expr, cpp_offset_expr

# Type hints
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.dispatcher import DefinedType

# Other imports
import itertools

WORMHOLE_STORAGE_TYPES = ["Wormhole_SRAM"]


class WormholeCodeGen(TargetCodeGenerator):
    """
    This is the new code generator for Wormhole.
    """

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher

        # Register array allocation/deallocation
        for dtype in WORMHOLE_STORAGE_TYPES:
            enum_type = dace.StorageType[dtype]
            self._dispatcher.register_array_dispatcher(enum_type, self)

        # Register copies to/from tensor cores
        # Is it needed?
        # gpu_storages = [
        #    dace.StorageType.GPU_Global,
        #    dace.StorageType.CPU_Pinned,
        #    dace.StorageType.GPU_Shared,
        #    dace.StorageType.Register,
        # ]
        # for src_storage, dst_storage in itertools.product(
        #    _TC_STORAGE_TYPES, gpu_storages
        # ):
        #    src_storage = dace.StorageType[src_storage]
        #    self._dispatcher.register_copy_dispatcher(
        #        src_storage, dst_storage, None, self
        #    )
        #    self._dispatcher.register_copy_dispatcher(
        #        dst_storage, src_storage, None, self
        #    )

    def allocate_array(
        self,
        sdfg: dace.SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Array,
        function_stream: CodeIOStream,
        declaration_stream: CodeIOStream,
        allocation_stream: CodeIOStream,
    ):
        pass
        # name = node.data

        ### Based on the hardware, the total size must be 16^2
        ##assert nodedesc.total_size == 16 * 16
        ### Majority is detected by the strides of the data
        ##maj = "row" if nodedesc.strides[-1] == 1 else "col"

        ## Write a fragment based on the storage type
        # if nodedesc.storage == dace.StorageType.TensorCore_Accumulator:
        #    ctype = "wmma::fragment<wmma::accumulator, 16, 16, 16, float>"
        #    declaration_stream.write(f"{ctype} {name};", cfg, state_id, node)
        # else:
        #    ctype = "wmma::fragment<wmma::matrix_{mat}, 16, 16, 16, half, wmma::{maj}_major>".format(
        #        mat=("a" if "A" in nodedesc.storage.name else "b"), maj=maj
        #    )
        #    declaration_stream.write(f"{ctype} {name};", cfg, state_id, node)

        ## Add the ctype to defined_vars so that the codegen can properly pass
        ## fragments to functions as an object reference.
        # self._dispatcher.defined_vars.add(name, DefinedType.Object, ctype)

    def deallocate_array(
        self,
        sdfg: dace.SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        node: nodes.AccessNode,
        nodedesc: dt.Array,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ):
        pass  # Nothing to deallocate (wmma::fragment is a C++ object)

    def copy_memory(
        self,
        sdfg: dace.SDFG,
        cfg: ControlFlowRegion,
        dfg: StateSubgraphView,
        state_id: int,
        src_node: nodes.Node,
        dst_node: nodes.Node,
        edge: MultiConnectorEdge,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        pass
        ## Obtain source and destination information, handle access<->tasklet
        ## If copying from tensor core fragments to/from tasklets, we only need
        ### to emit a reference, as the fragment contains the memory.
        # src_desc = (
        #    src_node.desc(sdfg) if isinstance(src_node, nodes.AccessNode) else None
        # )
        ## Tasklet -> Array
        # if not src_desc:
        #    local_name = dfg.memlet_path(edge)[0].src_conn
        #    callsite_stream.write(
        #        "auto& %s = %s;" % (local_name, dst_node.data),
        #        cfg,
        #        state_id,
        #        [src_node, dst_node],
        #    )
        #    return
        ##
        # dst_desc = (
        #    dst_node.desc(sdfg) if isinstance(dst_node, nodes.AccessNode) else None
        # )
        ## Array -> Tasklet
        # if not dst_desc:
        #    local_name = dfg.memlet_path(edge)[-1].dst_conn
        #    callsite_stream.write(
        #        "auto& %s = %s;" % (local_name, src_node.data),
        #        cfg,
        #        state_id,
        #        [src_node, dst_node],
        #    )
        #    return

        # nontc_desc = dst_desc if "TensorCore" in src_desc.storage.name else src_desc
        # nontc_node = dst_node if "TensorCore" in src_desc.storage.name else src_node

        ## Majority is detected by the strides of the data
        # row_major = True if nontc_desc.strides[-1] == 1 else False
        ######################################################################

        ## Set non-tensor-core C++ expression based on memlet
        # if edge.data.data == nontc_node.data:
        #    other_expr = cpp_array_expr(sdfg, edge.data)
        # elif edge.data.other_subset is not None:
        #    offset_cppstr = cpp_offset_expr(nontc_desc, edge.data.other_subset)
        #    other_expr = "%s[%s]" % (nontc_node.data, offset_cppstr)
        # else:
        #    other_expr = "%s[0]" % nontc_node.data
        ######################################################################

        ## Emit copy code
        # if "TensorCore" in dst_desc.storage.name:
        #    # GPU memory to Tensor Cores
        #    callsite_stream.write(
        #        "wmma::load_matrix_sync({tc}, &{other}, {stride});".format(
        #            tc=dst_node.data,
        #            other=other_expr,
        #            stride=src_desc.strides[0 if row_major else 1],
        #        ),
        #        cfg,
        #        state_id,
        #        [src_node, dst_node],
        #    )
        # else:
        #    # Tensor Cores to GPU memory
        #    callsite_stream.write(
        #        "wmma::store_matrix_sync(&{other}, {tc}, "
        #        "{stride}, wmma::mem_{maj}_major);".format(
        #            tc=src_node.data,
        #            other=other_expr,
        #            maj="row" if row_major else "col",
        #            stride=dst_desc.strides[0 if row_major else 1],
        #        ),
        #        cfg,
        #        state_id,
        #        [src_node, dst_node],
        #    )

    # def define_out_memlet(
    #    self,
    #    sdfg: dace.SDFG,
    #    cfg: ControlFlowRegion,
    #    dfg: StateSubgraphView,
    #    state_id: int,
    #    src_node: nodes.Node,
    #    dst_node: nodes.Node,
    #    edge: MultiConnectorEdge,
    #    function_stream: CodeIOStream,
    #    callsite_stream: CodeIOStream,
    # ):
    #    # Output memlets that are directed at WMMA fragments can use the "auto"
    #    # keyword for simplicity.
    #    callsite_stream.write(f"auto& {edge.src_conn} = {edge.data.data};")
