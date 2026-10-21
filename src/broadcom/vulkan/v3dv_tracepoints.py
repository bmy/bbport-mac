#
# Copyright © 2026 Igalia S.L.
# SPDX-License-Identifier: MIT
#

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--import-path', required=True)
    parser.add_argument('--utrace-src', required=True)
    parser.add_argument('--utrace-hdr', required=True)
    parser.add_argument('--perfetto-hdr', required=True)
    return parser.parse_args()


args = parse_args()
sys.path.insert(0, args.import_path)

from u_trace import ForwardDecl, Header, HeaderScope  # noqa: E402
from u_trace import Tracepoint  # noqa: E402
from u_trace import TracepointArg as Arg  # noqa: E402
from u_trace import TracepointArgStruct as ArgStruct  # noqa: E402
from u_trace import utrace_generate, utrace_generate_perfetto_utils  # noqa: E402

Header('vulkan/vulkan_core.h', scope=HeaderScope.HEADER)
ForwardDecl('struct v3dv_device')


def begin_end_tp(name, args=[], tp_struct=None):
    Tracepoint(
        f'begin_{name}',
        tp_perfetto=f'v3dv_utrace_perfetto_begin_{name}',
    )

    Tracepoint(
        f'end_{name}',
        args=args,
        tp_struct=tp_struct,
        tp_perfetto=f'v3dv_utrace_perfetto_end_{name}',
    )


def define_tracepoints():
    begin_end_tp(
        'job_cl',
        args=[
            Arg(type='uint32_t', var='id', c_format='%u'),
            Arg(type='uint8_t', var='serialize', c_format='0x%x'),
            Arg(type='uint32_t', var='draw_count', c_format='%u'),
            Arg(type='bool', var='cache_flush', c_format='%u')
        ]
    )

    begin_end_tp(
        'job_csd',
        args=[
            Arg(type='uint32_t', var='id', c_format='%u'),
            Arg(type='uint8_t', var='serialize', c_format='0x%x'),
            Arg(type='uint32_t', var='wg_x', c_format='%u'),
            Arg(type='uint32_t', var='wg_y', c_format='%u'),
            Arg(type='uint32_t', var='wg_z', c_format='%u')
        ]
    )

    begin_end_tp(
        'job_tfu',
        args=[
            Arg(type='uint32_t', var='id', c_format='%u'),
            Arg(type='uint8_t', var='serialize', c_format='0x%x')
        ]
    )

    begin_end_tp(
        'job_cpu_reset_queries',
        args=[
            Arg(type='uint32_t', var='id', c_format='%u'),
            Arg(type='uint8_t', var='serialize', c_format='0x%x')
        ]
    )

    begin_end_tp(
        'job_cpu_copy_query_results',
        args=[
            Arg(type='uint32_t', var='id', c_format='%u'),
            Arg(type='uint8_t', var='serialize', c_format='0x%x')
        ]
    )

    begin_end_tp(
        'job_cpu_csd_indirect',
        args=[
            Arg(type='uint32_t', var='id', c_format='%u'),
            Arg(type='uint8_t', var='serialize', c_format='0x%x')
        ]
    )

    begin_end_tp(
        'job_cpu_timestamp_query',
        args=[
            Arg(type='uint32_t', var='id', c_format='%u'),
            Arg(type='uint8_t', var='serialize', c_format='0x%x')
        ]
    )

    begin_end_tp(
        'cmdbuf',
        args=[
            Arg(
                type='VkCommandBufferUsageFlags',
                var='flags',
                c_format='0x%x',
            )
        ]
    )


def generate_code():
    utrace_generate(
        cpath=args.utrace_src,
        hpath=args.utrace_hdr,
        ctx_param='struct v3dv_device *dev',
    )

    utrace_generate_perfetto_utils(hpath=args.perfetto_hdr)


def main():
    define_tracepoints()
    generate_code()


if __name__ == '__main__':
    main()
