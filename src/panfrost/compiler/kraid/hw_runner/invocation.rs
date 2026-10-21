use std::borrow::Borrow;
use std::ffi::c_void;
use std::io;
use std::ptr::null_mut;
use std::sync::Arc;

use crate::device::{MemoryBuffer, VirtualMemory};
use kraid_hw_runner_bindings::{
    hw_runner_buffer_descr, hw_runner_invocation_info, hw_runner_layout_info,
    hw_runner_new_cmd_stream, hw_runner_shader_args,
};

pub struct InvocationInfo<'a> {
    pub code: &'a [u8],
    pub register_count: u8,
    pub register_preload: u64,
    pub invocations: u32,
    pub fau: &'a [u32],
    // Offset of ShaderArgs in fau
    pub fau_args_offset: usize,
    pub data: &'a mut [u8],
    pub data_stride: u32,
    // PKA buffers (for BufferDescriptors)
    pub buffers: &'a mut [&'a mut [u8]],
}

impl<'a> InvocationInfo<'a> {
    /// Assigns the layout of RW data buffers-padding
    fn assign_offsets<B: Borrow<[u8]>>() -> impl FnMut(B) -> (u64, B) + use<B> {
        let mut offset_B = 0u64;

        // Packed together, padding inserted where necessary to keep 16-byte
        // alignment
        move |buf| {
            let off = offset_B.next_multiple_of(16);
            offset_B = off + buf.borrow().len() as u64;
            (off, buf)
        }
    }

    /// Returns a read-only view of rw-data with its layout
    fn rw_data_layout(
        &self,
    ) -> impl Iterator<Item = (u64, &[u8])> + use<'_, 'a> {
        std::iter::once(&*self.data)
            .chain(self.buffers.iter().map(|b| &**b))
            .map(Self::assign_offsets())
    }

    /// Returns a mutable view of rw-data with its layout
    fn rw_data_layout_mut(
        &mut self,
    ) -> impl Iterator<Item = (u64, &mut [u8])> + use<'_, 'a> {
        std::iter::once(&mut *self.data)
            .chain(self.buffers.iter_mut().map(|b| &mut **b))
            .map(Self::assign_offsets())
    }

    fn rw_data_size_B(&self) -> u64 {
        self.rw_data_layout()
            .last()
            .map(|(off, buf)| off + buf.len() as u64)
            .unwrap_or(0)
    }

    fn copy_rw_to(&self, dest: &mut [u8]) {
        for (off, buf) in self.rw_data_layout() {
            let off = usize::try_from(off).unwrap();
            dest[off..(off + buf.len())].copy_from_slice(buf);
        }
    }

    fn copy_rw_from(&mut self, src: &[u8]) {
        for (off, buf) in self.rw_data_layout_mut() {
            let off = usize::try_from(off).unwrap();
            buf.copy_from_slice(&src[off..(off + buf.len())]);
        }
    }
}

pub struct InvocationCmdStream {
    pub descr_buf: MemoryBuffer,
    pub data_buf: MemoryBuffer,
    // Offset of the command stream to run
    pub cs_offset: u64,
    pub cs_len: u64,
}

impl InvocationCmdStream {
    pub fn cs_device_addr(&self) -> u64 {
        self.descr_buf.device_addr() + self.cs_offset
    }

    pub fn copy_back(&self, mut invoc: InvocationInfo) {
        self.data_buf.sync();
        invoc.copy_rw_from(self.data_buf.data_view());
    }
}

pub fn new_invocation_cs(
    mem: &Arc<VirtualMemory>,
    info: &InvocationInfo,
) -> io::Result<InvocationCmdStream> {
    const SHADER_ARGS_FAU_ENTRIES: usize =
        size_of::<hw_runner_shader_args>() / size_of::<u32>();
    assert!(info.fau_args_offset + SHADER_ARGS_FAU_ENTRIES <= info.fau.len());
    assert!((info.data_stride * info.invocations) as usize <= info.data.len());

    let mut data_buf = mem.allocate_buffer(
        info.rw_data_size_B(),
        c"hw_runner writable data",
        0,
    )?;

    // RW layout
    let data_stride = info.data_stride;
    let mut rw_layout = info.rw_data_layout();
    let (data_off, _buf) = rw_layout.next().unwrap();

    let shader_args = hw_runner_shader_args {
        data_addr: data_buf.device_addr() + data_off,
        data_stride,
        _pad: 0,
    };

    let mut buffer_descrs: Vec<_> = rw_layout
        .map(|(off, buf)| hw_runner_buffer_descr {
            device_ptr: data_buf.device_addr() + off,
            size_B: buf.len().try_into().expect("Buffer too large"),
        })
        .collect();
    assert_eq!(buffer_descrs.len(), info.buffers.len());

    let mut invoc_data = hw_runner_invocation_info {
        // Initialized later
        descr_bo_device_ptr: 0,
        descr_bo_host_ptr: null_mut(),

        code_ptr: info.code.as_ptr() as *mut c_void,
        code_size_B: info.code.len() as u64,
        fau_ptr: info.fau.as_ptr() as *mut c_void,
        fau_size_B: size_of_val(info.fau) as u64,
        args_fau_offset: info.fau_args_offset as u64,
        shader_args,

        simple_buffers: buffer_descrs.as_mut_ptr(),
        simple_buffer_count: info.buffers.len() as u32,

        register_preload: info.register_preload,
        register_count: info.register_count,
        invocations: info.invocations,
    };

    let mut layout_info = hw_runner_layout_info {
        descr_bo_size_B: 0,
        cs_offset: 0,
        cs_size_B: 0,
    };
    unsafe {
        hw_runner_new_cmd_stream(
            mem.dev().kdev(),
            &mut invoc_data,
            &mut layout_info,
        )
    };

    let descr_buf = mem.allocate_buffer(
        layout_info.descr_bo_size_B,
        c"hw_runner invoc data",
        1,
    )?;
    invoc_data.descr_bo_device_ptr = descr_buf.device_addr();
    invoc_data.descr_bo_host_ptr = descr_buf.host_addr();

    unsafe {
        hw_runner_new_cmd_stream(
            mem.dev().kdev(),
            &mut invoc_data,
            &mut layout_info,
        )
    };

    descr_buf.sync();
    // Copy RW data and sync
    info.copy_rw_to(data_buf.data_view_mut());
    data_buf.sync();

    Ok(InvocationCmdStream {
        descr_buf,
        data_buf,
        cs_offset: layout_info.cs_offset,
        cs_len: layout_info.cs_size_B,
    })
}
