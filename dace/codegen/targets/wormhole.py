# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

# General DaCe imports
import dace
from dace import data as dt
from dace.sdfg import nodes, SDFG
from dace import registry, dtypes, data
from dace import memlet as mmlt
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.sdfg import (
    ScopeSubgraphView,
    SDFG,
    scope_contains_scope,
    is_array_stream_view,
    NodeNotExpandedError,
    dynamic_map_inputs,
)
from typing import Union
from dace.codegen.codeobject import CodeObject
from dace.codegen.prettycode import CodeIOStream

# Code generator imports and helpers
from dace.codegen.targets.framecode import DaCeCodeGenerator
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.cpp import cpp_array_expr, cpp_offset_expr
from dace.codegen.targets import cpp
from dace.codegen.targets.cpp import sym2cpp

# Type hints
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.dispatcher import DefinedType

# Other imports
import itertools

WORMHOLE_STORAGE_TYPES = [dace.StorageType.Wormhole_SRAM]


@registry.autoregister_params(name="wormhole")
class WormholeCodeGen(TargetCodeGenerator):
    """
    This is the new code generator for Wormhole.
    """

    target_name = "wormhole"

    def __init__(self, frame_codegen: DaCeCodeGenerator, sdfg: dace.SDFG):
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._global_sdfg: SDFG = sdfg

        self._program_name = sdfg.name
        self._kernel_codes = []

        # Register array allocation/deallocation
        for dtype in WORMHOLE_STORAGE_TYPES:
            self._dispatcher.register_array_dispatcher(dtype, self)

        self._dispatcher.register_map_dispatcher(
            dtypes.ScheduleType.Wormhole_Kernel, self
        )

        self._cpu_codegen = self._dispatcher.get_generic_node_dispatcher()

        # We need it to get all kernels with Wormhole schedule
        # Each one becomes a separate kernel
        # self._dispatcher.register_map_dispatcher(
        #    [dtypes.ScheduleType.Wormhole_Kernel], self
        # )

        cpu_storages = [
            dace.StorageType.CPU_Pinned,
            dace.StorageType.CPU_Heap,
            dace.StorageType.Register,
        ]
        for src_storage, dst_storage in itertools.product(
            [dace.StorageType.Wormhole_SRAM], cpu_storages
        ):
            self._dispatcher.register_copy_dispatcher(
                src_storage, dst_storage, None, self
            )
            self._dispatcher.register_copy_dispatcher(
                dst_storage, src_storage, None, self
            )

    def generate_scope(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg_scope: ScopeSubgraphView,
        state_id: int,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        entry_node = dfg_scope.source_nodes()[0]

        assert isinstance(entry_node, nodes.MapEntry)

        if sdfg.parent_nsdfg_node is not None:
            kernel_name = (
                f"{sdfg.parent_nsdfg_node.label}_{entry_node.label}_{cfg.cfg_id}"
            )
        else:
            kernel_name = f"{entry_node.label}_{cfg.cfg_id}"

        kernel_stream = CodeIOStream()

        # kernel_stream.write("{", cfg, state_id, entry_node)

        for param, rng in zip(entry_node.map.params, entry_node.map.range):
            # We use the sym2cpp function from the cpp support functions
            # to convert symbolic expressions to proper C++
            begin, end, stride = (sym2cpp(r) for r in rng)
            end = sym2cpp(rng[1] + 1)

            # Every write is optionally (but recommended to be) tagged with
            # 1-3 extra arguments, serving as line information to match
            # SDFG, state, and graph nodes/edges to written code.
            kernel_stream.write(
                f"""// Loopy-loop {param}
            for (int {param} = {begin}; {param} < {end}; {param} += {stride}) {{""",
                sdfg,
                state_id,
                entry_node,
            )

        self.generate_node(
            sdfg, cfg, dfg_scope, state_id, entry_node, function_stream, kernel_stream
        )

        self._dispatcher.dispatch_subgraph(
            sdfg,
            cfg,
            dfg_scope,
            state_id,
            function_stream,
            kernel_stream,
            skip_entry_node=True,
        )

        # Write kernel prototype
        localcode = CodeIOStream()
        localcode.write("namespace NAMESPACE {")
        localcode.write("void MAIN {")

        self._frame.generate_constants(sdfg, localcode)

        localcode.write(kernel_stream.getvalue() + "\n")

        # hack, no idea why braces are missing from the loop
        localcode.write("}", cfg, state_id, entry_node)
        localcode.write("}", cfg, state_id, entry_node)
        localcode.write("}", cfg, state_id, entry_node)

        self._kernel_codes.append((kernel_name, localcode.getvalue()))

    # Copied from CPU
    def generate_node(
        self,
        sdfg: SDFG,
        cfg: ControlFlowRegion,
        dfg: ScopeSubgraphView,
        state_id: int,
        node: nodes.Node,
        function_stream: CodeIOStream,
        callsite_stream: CodeIOStream,
    ) -> None:
        ## Dynamically obtain node generator according to class name
        # try:
        #    gen = getattr(self, "_generate_" + type(node).__name__)
        # except AttributeError:
        #    if isinstance(node, nodes.LibraryNode):
        #        raise NodeNotExpandedError(sdfg, state_id, dfg.node_id(node))
        #    raise

        # gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

        ## Mark node as "generated"
        # self._generated_nodes.add(node)
        # self._locals.clear_scope(self._ldepth + 1)
        self._cpu_codegen.generate_node(
            sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream
        )

    def get_generated_codeobjects(self):
        host_code = CodeIOStream()
        host_code.write("""
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>

using namespace tt;
using namespace tt::tt_metal;

""")
        host_code.write("\n\n")

        # FIXME: pass
        self._frame.generate_fileheader(self._global_sdfg, host_code, "wormhole")

        params_comma = self._global_sdfg.init_signature(
            free_symbols=self._frame.free_symbols(self._global_sdfg)
        )
        if params_comma:
            params_comma = ", " + params_comma

        host_code_obj = CodeObject(
            self._program_name,
            host_code.getvalue(),
            "cpp",
            WormholeCodeGen,
            "Wormhhole",
            target_type="host",
            sdfg=self._global_sdfg,
        )

        kernel_code_objs = [
            CodeObject(
                kernel_file_name,
                kernel_code,
                "cl",
                WormholeCodeGen,
                "Wormhhole",
                target_type="device",
                sdfg=self._global_sdfg,
            )
            for (kernel_file_name, kernel_code) in self._kernel_codes
        ]

        return [host_code_obj] + kernel_code_objs

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
        # print('alloc_array', name)

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
        name = node.data
        self._dispatcher.defined_vars.add(
            name, DefinedType.Object, nodedesc.dtype.ctype
        )

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
        # print("deallocate", node.name)
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

    def define_out_memlet(
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
    ):
        ctype = edge.src.out_connectors[edge.src_conn].ctype
        self._dispatcher.defined_vars.add(edge.src_conn, DefinedType.Scalar, ctype)
        callsite_stream.write(f"auto& {edge.src_conn} = {edge.data.data};")
