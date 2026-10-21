# Copyright © 2026 Imagination Technologies Ltd.
# SPDX-License-Identifier: MIT
import argparse
import sys

# List of the default tracepoints enabled. By default most tracepoints are
# enabled, set tp_default_enabled=False to disable them by default.
#
pvr_default_tps = []

#
# Tracepoint definitions:
#
def define_tracepoints(args):
    from u_trace import Header, HeaderScope
    from u_trace import ForwardDecl
    from u_trace import Tracepoint
    from u_trace import TracepointArg as Arg
    from u_trace import TracepointArgStruct as ArgStruct

    Header('inttypes.h', scope=HeaderScope.SOURCE)
    Header('pvr_driver_ds.h', scope=HeaderScope.SOURCE)
    Header('vulkan/vulkan_core.h', scope=HeaderScope.SOURCE)
    Header('ds/pvr_driver_ds.h', scope=HeaderScope.HEADER)

    def begin_end_tp(name, tp_args=[], tp_struct=None, tp_print=None,
                     tp_default_enabled=True, end_pipelined=True,
                     compute=False, repeat_last=False,
                     need_cs_param=False,
                     toggle_name=None):
        global pvr_default_tps
        if toggle_name is None:
            toggle_name = name
        if tp_default_enabled and toggle_name not in pvr_default_tps:
            pvr_default_tps.append(toggle_name)

        Tracepoint('pvr_begin_{0}'.format(name),
                   toggle_name=toggle_name,
                   args=tp_args,
                   tp_markers='pvr_ds_begin_{0}'.format(name),
                   need_cs_param=need_cs_param)

        tp_flags = []
        if end_pipelined:
            if compute:
                tp_flags.append('PVR_DS_TRACEPOINT_FLAG_END_CS')
            elif repeat_last:
                tp_flags.append('PVR_DS_TRACEPOINT_FLAG_REPEAT_LAST')
            else:
                tp_flags.append('PVR_DS_TRACEPOINT_FLAG_END_OF_PIPE')

        Tracepoint('pvr_end_{0}'.format(name),
                   toggle_name=toggle_name,
                   tp_struct=tp_struct,
                   tp_markers='pvr_ds_end_{0}'.format(name),
                   tp_print=tp_print,
                   tp_flags=tp_flags,
                   need_cs_param=need_cs_param)

    # obj_id is (uintptr_t)&cmd_buffer->vk.base — the vk_object_base address,
    obj_id_arg = Arg(type='uintptr_t', var='obj_id', c_format='0x%" PRIxPTR "')

    # Compute workload events are traced
    begin_end_tp('compute',
                 tp_args=[obj_id_arg],
                 need_cs_param=True,
                 compute=True)

    # Graphics (Geom/Frag) workload events
    begin_end_tp('draw',
                 tp_args=[obj_id_arg,
                          Arg(type='enum pvr_draw_op', var='op', c_format='%s', to_prim_type='pvr_draw_op_to_str({})'),],
                 need_cs_param=True)

    # Transfer workload events
    begin_end_tp('transfer',
                 tp_args=[obj_id_arg,
                          Arg(type='enum pvr_transfer_op', var='op', c_format='%s', to_prim_type='pvr_transfer_op_to_str({})'),],
                 need_cs_param=True)


def generate_code(args):
    from u_trace import utrace_generate
    from u_trace import TRACEPOINTS

    # Disable print/print_json generation for all PVR tracepoints.
    for tp in TRACEPOINTS.values():
        tp.can_generate_print = lambda: False

    utrace_generate(cpath=args.utrace_src, hpath=args.utrace_hdr,
                    ctx_param='struct pvr_cmd_buffer *cmd_buffer',
                    trace_toggle_name='pvr_gpu_tracepoint',
                    trace_toggle_defaults=pvr_default_tps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--import-path', required=True)
    parser.add_argument('--utrace-src', required=True)
    parser.add_argument('--utrace-hdr', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.import_path)
    define_tracepoints(args)
    generate_code(args)


if __name__ == '__main__':
    main()
