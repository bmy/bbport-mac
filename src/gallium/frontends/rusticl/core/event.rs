// Copyright 2020 Red Hat.
// SPDX-License-Identifier: MIT

use crate::api::icd::*;
use crate::api::types::*;
use crate::core::context::*;
use crate::core::device::*;
use crate::core::queue::*;
use crate::impl_cl_type_trait;

use mesa_rust::pipe::context::RWFlags;
use mesa_rust::pipe::query::*;
use mesa_rust::pipe::resource::PipeResourceOwned;
use mesa_rust::pipe::screen::ResourceType;
use mesa_rust_gen::*;
use mesa_rust_util::static_assert;
use rusticl_opencl_gen::*;

use std::collections::HashSet;
use std::mem;
use std::sync::atomic;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::Weak;
use std::time::Duration;

// we assert that those are a continous range of numbers so we won't have to use HashMaps
static_assert!(CL_COMPLETE == 0);
static_assert!(CL_RUNNING == 1);
static_assert!(CL_SUBMITTED == 2);
static_assert!(CL_QUEUED == 3);

pub type EventSig =
    Box<dyn FnOnce(&Context, &mut QueueContextWithState) -> CLResult<()> + Send + Sync>;

const PROFILING_ORDERING: atomic::Ordering = atomic::Ordering::SeqCst;

pub enum EventTimes {
    Queued = CL_PROFILING_COMMAND_QUEUED as isize,
    Submit = CL_PROFILING_COMMAND_SUBMIT as isize,
    Start = CL_PROFILING_COMMAND_START as isize,
    End = CL_PROFILING_COMMAND_END as isize,
}

#[derive(Default)]
struct EventMutState {
    status: cl_int,
    cbs: [Vec<EventCB>; 3],
    work: Option<EventSig>,
}

trait Profiler: Send + Sync {
    fn profile_work(
        &self,
        cl_ctx: &Context,
        ctx: &mut QueueContextWithState,
        work: Option<EventSig>,
    ) -> CLResult<()>;
    fn get_time(&self, which: EventTimes) -> cl_ulong;
    fn mark_queued(&self, time: cl_ulong);
}

struct DisabledProfiler {}

impl Profiler for DisabledProfiler {
    fn get_time(&self, _which: EventTimes) -> cl_ulong {
        0
    }

    fn mark_queued(&self, _time: cl_ulong) {}

    fn profile_work(
        &self,
        cl_ctx: &Context,
        ctx: &mut QueueContextWithState,
        work: Option<EventSig>,
    ) -> CLResult<()> {
        if let Some(w) = work {
            w(cl_ctx, ctx)
        } else {
            Ok(())
        }
    }
}

#[derive(Default)]
struct CPUProfiler {
    time_queued: AtomicU64,
    time_submit: AtomicU64,
    time_start: AtomicU64,
    time_end: AtomicU64,
}

impl CPUProfiler {
    fn set_time(&self, which: EventTimes, value: cl_ulong) {
        match which {
            EventTimes::Queued => &self.time_queued.store(value, PROFILING_ORDERING),
            EventTimes::Submit => &self.time_submit.store(value, PROFILING_ORDERING),
            EventTimes::Start => &self.time_start.store(value, PROFILING_ORDERING),
            EventTimes::End => &self.time_end.store(value, PROFILING_ORDERING),
        };
    }
}

impl Profiler for CPUProfiler {
    fn profile_work(
        &self,
        cl_ctx: &Context,
        ctx: &mut QueueContextWithState,
        work: Option<EventSig>,
    ) -> CLResult<()> {
        self.set_time(EventTimes::Submit, ctx.dev.screen().get_timestamp());
        if let Some(w) = work {
            let mut query_start =
                PipeQueryGen::<{ pipe_query_type::PIPE_QUERY_TIMESTAMP }>::new(ctx.ctx)
                    .ok_or(CL_OUT_OF_HOST_MEMORY)?;
            w(cl_ctx, ctx)?;
            let mut query_end =
                PipeQueryGen::<{ pipe_query_type::PIPE_QUERY_TIMESTAMP }>::new(ctx.ctx)
                    .ok_or(CL_OUT_OF_HOST_MEMORY)?;

            self.set_time(EventTimes::Start, query_start.read_blocked());
            self.set_time(EventTimes::End, query_end.read_blocked());
        }

        Ok(())
    }

    fn get_time(&self, which: EventTimes) -> cl_ulong {
        match which {
            EventTimes::Queued => self.time_queued.load(PROFILING_ORDERING),
            EventTimes::Submit => self.time_submit.load(PROFILING_ORDERING),
            EventTimes::Start => self.time_start.load(PROFILING_ORDERING),
            EventTimes::End => self.time_end.load(PROFILING_ORDERING),
        }
    }

    fn mark_queued(&self, time: cl_ulong) {
        self.set_time(EventTimes::Queued, time);
    }
}

struct GPUProfiler {
    time_queued: AtomicU64,
    time_submit: AtomicU64,
    res: PipeResourceOwned,
    dev: &'static Device,
}

impl GPUProfiler {
    fn new(dev: &'static Device) -> CLResult<Self> {
        let res = dev
            .screen()
            .resource_create_buffer(
                0x10,
                ResourceType::Staging,
                PIPE_BIND_QUERY_BUFFER,
                0,
                // PIPE_RESOURCE_FLAG_MAP_COHERENT | PIPE_RESOURCE_FLAG_MAP_PERSISTENT,
            )
            .ok_or(CL_OUT_OF_RESOURCES)?;

        Ok(Self {
            time_queued: 0.into(),
            time_submit: 0.into(),
            dev: dev,
            res: res,
        })
    }

    fn has_timestamp_query_raw(&self) -> bool {
        self.dev.screen().is_convert_timestamp_supported()
    }

    fn read_timestamp_query(&self, offset: i32) -> cl_ulong {
        let helper_ctx = self.dev.helper_ctx();
        let tx = helper_ctx
            .buffer_map(&self.res, offset, 8, RWFlags::RD)
            .unwrap();
        let mut ts = unsafe { tx.ptr().cast::<cl_ulong>().read() };
        if self.has_timestamp_query_raw() {
            ts = self.dev.screen().convert_timestamp(ts);
        }
        ts
    }

    fn create_timestamp_query<'c>(&self, ctx: &'c QueueContext) -> CLResult<PipeQuery<'c, u64>> {
        let query_start = if self.has_timestamp_query_raw() {
            PipeQueryGen::<{ pipe_query_type::PIPE_QUERY_TIMESTAMP_RAW }>::new(ctx)
        } else {
            PipeQueryGen::<{ pipe_query_type::PIPE_QUERY_TIMESTAMP }>::new(ctx)
        };

        query_start.ok_or(CL_OUT_OF_HOST_MEMORY)
    }
}

impl Profiler for GPUProfiler {
    fn profile_work(
        &self,
        cl_ctx: &Context,
        ctx: &mut QueueContextWithState,
        work: Option<EventSig>,
    ) -> CLResult<()> {
        self.time_submit
            .store(ctx.dev.screen().get_timestamp(), PROFILING_ORDERING);

        if let Some(w) = work {
            let mut query_start = self.create_timestamp_query(ctx.ctx)?;
            query_start.write_to_resource(&self.res, 0);

            w(cl_ctx, ctx)?;

            let mut query_end = self.create_timestamp_query(ctx.ctx)?;
            query_end.write_to_resource(&self.res, 8);
        }

        Ok(())
    }

    fn get_time(&self, which: EventTimes) -> cl_ulong {
        match which {
            EventTimes::Queued => self.time_queued.load(PROFILING_ORDERING),
            EventTimes::Submit => self.time_submit.load(PROFILING_ORDERING),
            EventTimes::Start => self.read_timestamp_query(0),
            EventTimes::End => self.read_timestamp_query(8),
        }
    }

    fn mark_queued(&self, time: cl_ulong) {
        self.time_queued.store(time, PROFILING_ORDERING);
    }
}

pub struct GPUEvent {
    cmd_type: cl_command_type,
    queue: Weak<Queue>,
    deps: Vec<Arc<Event>>,
    profiling: Box<dyn Profiler>,
}

impl GPUEvent {
    pub fn get_time(&self, which: EventTimes) -> cl_ulong {
        self.profiling.get_time(which)
    }
}

enum EventImpl {
    UserEvent,
    GPUEvent(GPUEvent),
}

pub struct Event {
    pub base: CLObjectBase<CL_INVALID_EVENT>,
    pub context: Arc<Context>,
    state: Mutex<EventMutState>,
    cv: Condvar,
    kind: EventImpl,
}

impl_cl_type_trait!(cl_event, Event, CL_INVALID_EVENT);

impl Event {
    pub fn new(
        queue: &Arc<Queue>,
        cmd_type: cl_command_type,
        deps: Vec<Arc<Event>>,
        work: EventSig,
    ) -> CLResult<Arc<Event>> {
        let profiler: Box<dyn Profiler> = if queue.is_profiling_enabled() {
            if queue.device.screen().caps().query_buffer_object {
                Box::new(GPUProfiler::new(queue.device)?)
            } else {
                Box::new(CPUProfiler::default())
            }
        } else {
            Box::new(DisabledProfiler {})
        };

        Ok(Arc::new(Self {
            base: CLObjectBase::new(RusticlTypes::Event),
            context: Arc::clone(&queue.context),
            state: Mutex::new(EventMutState {
                status: CL_QUEUED as cl_int,
                work: Some(work),
                ..Default::default()
            }),
            kind: EventImpl::GPUEvent(GPUEvent {
                cmd_type: cmd_type,
                queue: Arc::downgrade(queue),
                deps: deps,
                profiling: profiler,
            }),
            cv: Condvar::new(),
        }))
    }

    pub fn new_user(context: Arc<Context>) -> Arc<Event> {
        Arc::new(Self {
            base: CLObjectBase::new(RusticlTypes::Event),
            context: context,
            state: Mutex::new(EventMutState {
                status: CL_SUBMITTED as cl_int,
                ..Default::default()
            }),
            kind: EventImpl::UserEvent,
            cv: Condvar::new(),
        })
    }

    fn state(&self) -> MutexGuard<'_, EventMutState> {
        self.state.lock().unwrap()
    }

    pub fn status(&self) -> cl_int {
        self.state().status
    }

    fn set_status(&self, mut lock: MutexGuard<EventMutState>, new: cl_int) {
        lock.status = new;

        // signal on completion or an error
        if new <= CL_COMPLETE as cl_int {
            self.cv.notify_all();
        }

        // errors we treat as CL_COMPLETE
        let cb_max = if new < 0 { CL_COMPLETE } else { new as u32 };

        // there are only callbacks for those
        if ![CL_COMPLETE, CL_RUNNING, CL_SUBMITTED].contains(&cb_max) {
            return;
        }

        let mut cbs = Vec::new();
        // Collect all cbs we need to call first. Technically it is not required to call them in
        // order, but let's be nice to applications as it's for free
        for idx in (cb_max..=CL_SUBMITTED).rev() {
            cbs.extend(
                // use mem::take as each callback is only supposed to be called exactly once
                mem::take(&mut lock.cbs[idx as usize])
                    .into_iter()
                    // we need to save the status this cb was registered with
                    .map(|cb| (idx as cl_int, cb)),
            );
        }

        // applications might want to access the event in the callback, so drop the lock before
        // calling into the callbacks.
        drop(lock);

        for (idx, cb) in cbs {
            // from the CL spec:
            //
            // event_command_status is equal to the command_exec_callback_type used while
            // registering the callback. [...] If the callback is called as the result of the
            // command associated with event being abnormally terminated, an appropriate error code
            // for the error that caused the termination will be passed to event_command_status
            // instead.
            let status = if new < 0 { new } else { idx };
            cb.call(self, status);
        }
    }

    pub fn set_user_status(&self, status: cl_int) {
        self.set_status(self.state(), status);
    }

    pub fn is_error(&self) -> bool {
        self.status() < 0
    }

    pub fn is_user(&self) -> bool {
        matches!(self.kind, EventImpl::UserEvent)
    }

    pub fn mark_queued(&self, value: cl_ulong) {
        match &self.kind {
            EventImpl::GPUEvent(gpu) => gpu.profiling.mark_queued(value),
            EventImpl::UserEvent => {}
        }
    }

    pub fn add_cb(&self, state: cl_int, cb: EventCB) {
        let mut lock = self.state();
        let status = lock.status;

        // call cb if the status was already reached
        if state >= status {
            drop(lock);
            cb.call(self, state);
        } else {
            lock.cbs.get_mut(state as usize).unwrap().push(cb);
        }
    }

    pub(super) fn signal(&self) {
        let state = self.state();
        // we don't want to call signal on errored events, but if that still happens, handle it
        // gracefully
        debug_assert_eq!(state.status, CL_SUBMITTED as cl_int);
        if state.status < 0 {
            return;
        }
        self.set_status(state, CL_RUNNING as cl_int);
        self.set_status(self.state(), CL_COMPLETE as cl_int);
    }

    pub fn wait(&self) -> cl_int {
        let mut lock = self.state();
        while lock.status >= CL_RUNNING as cl_int {
            lock = self
                .cv
                .wait_timeout(lock, Duration::from_secs(1))
                .unwrap()
                .0;
        }
        lock.status
    }

    fn call_inner(&self, ctx: &mut QueueContextWithState, work: Option<EventSig>) -> CLResult<()> {
        let gpu = self.gpu_event().unwrap();
        gpu.profiling.profile_work(&self.context, ctx, work)
    }

    // We always assume that work here simply submits stuff to the hardware even if it's just doing
    // sw emulation or nothing at all.
    // If anything requets waiting, we will update the status through fencing later.
    pub fn call(&self, ctx: &mut QueueContextWithState) -> cl_int {
        let mut lock = self.state();
        let mut status = lock.status;

        if status == CL_QUEUED as cl_int {
            status = self
                .call_inner(ctx, lock.work.take())
                .err()
                .unwrap_or(CL_SUBMITTED as cl_int);
            self.set_status(lock, status);
        }
        status
    }

    fn deep_unflushed_deps_impl<'a>(&'a self, result: &mut HashSet<&'a Event>) {
        if self.status() <= CL_SUBMITTED as i32 {
            return;
        }

        // only scan dependencies if it's a new one
        if result.insert(self) {
            for e in self.deps() {
                e.deep_unflushed_deps_impl(result);
            }
        }
    }

    /// does a deep search and returns a list of all dependencies including `events` which haven't
    /// been flushed out yet
    pub fn deep_unflushed_deps(events: &[Arc<Event>]) -> HashSet<&Event> {
        let mut result = HashSet::new();

        for e in events {
            e.deep_unflushed_deps_impl(&mut result);
        }

        result
    }

    /// does a deep search and returns a list of all queues which haven't been flushed yet
    pub fn deep_unflushed_queues(events: &[Arc<Event>]) -> HashSet<Arc<Queue>> {
        Event::deep_unflushed_deps(events)
            .iter()
            .filter_map(|e| e.queue())
            // We don't have to do anything for destroyed queues as they already flush on drop.
            .filter_map(Weak::upgrade)
            .collect()
    }

    pub fn cmd_type(&self) -> cl_command_type {
        match &self.kind {
            EventImpl::GPUEvent(gpu) => gpu.cmd_type,
            EventImpl::UserEvent => CL_COMMAND_USER,
        }
    }

    pub fn queue(&self) -> Option<&Weak<Queue>> {
        match &self.kind {
            EventImpl::GPUEvent(gpu) => Some(&gpu.queue),
            EventImpl::UserEvent => None,
        }
    }

    pub fn deps(&self) -> &[Arc<Event>] {
        match &self.kind {
            EventImpl::GPUEvent(gpu) => &gpu.deps,
            EventImpl::UserEvent => &[],
        }
    }

    pub fn gpu_event(&self) -> Option<&GPUEvent> {
        match &self.kind {
            EventImpl::GPUEvent(gpu) => Some(gpu),
            EventImpl::UserEvent => None,
        }
    }
}

impl Drop for Event {
    // implement drop in order to prevent stack overflows of long dependency chains.
    //
    // This abuses the fact that `Arc::into_inner` only succeeds when there is one strong reference
    // so we turn a recursive drop chain into a drop list for events having no other references.
    fn drop(&mut self) {
        let EventImpl::GPUEvent(gpu) = &mut self.kind else {
            return;
        };

        let mut deps_list = vec![mem::take(&mut gpu.deps)];
        while let Some(deps) = deps_list.pop() {
            for dep in deps {
                if let Some(mut dep) = Arc::into_inner(dep) {
                    let EventImpl::GPUEvent(ref mut gpu) = dep.kind else {
                        continue;
                    };

                    deps_list.push(mem::take(&mut gpu.deps));
                }
            }
        }
    }
}

// TODO worker thread per device
// Condvar to wait on new events to work on
// notify condvar when flushing queue events to worker
// attach fence to flushed events on context->flush
// store "newest" event for in-order queues per queue
// reordering/graph building done in worker
