// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::future::Either;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::{HostAllocatorRef, WritableHostBuffer};
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

/// Optional bounded hedging for object-store range reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectStoreReadHedgeConfig {
    pub delay: Duration,
    pub min_bytes: usize,
    pub max_bytes: usize,
}

impl ObjectStoreReadHedgeConfig {
    fn applies_to(self, length: usize) -> bool {
        self.delay > Duration::ZERO && length >= self.min_bytes && length <= self.max_bytes
    }
}

/// Physical object-store request counters for `ObjectStoreReadAt`.
///
/// `VortexReadAt::read_at` is a logical range-read API. Tail hedging can issue
/// more than one physical GET for one logical read, so callers that tune object
/// serving need counters below the logical abstraction.
#[derive(Default, Debug)]
pub struct ObjectStoreReadStats {
    requests_started: AtomicU64,
    requests_completed: AtomicU64,
    bytes_started: AtomicU64,
    max_bytes_started: AtomicU64,
    hedge_requests_started: AtomicU64,
    hedge_wins: AtomicU64,
    total_completed_nanos: AtomicU64,
    max_completed_nanos: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ObjectStoreReadStatsSnapshot {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub bytes_started: u64,
    pub max_bytes_started: u64,
    pub hedge_requests_started: u64,
    pub hedge_wins: u64,
    pub total_completed_nanos: u64,
    pub max_completed_nanos: u64,
}

impl ObjectStoreReadStats {
    pub fn snapshot(&self) -> ObjectStoreReadStatsSnapshot {
        ObjectStoreReadStatsSnapshot {
            requests_started: self.requests_started.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            bytes_started: self.bytes_started.load(Ordering::Relaxed),
            max_bytes_started: self.max_bytes_started.load(Ordering::Relaxed),
            hedge_requests_started: self.hedge_requests_started.load(Ordering::Relaxed),
            hedge_wins: self.hedge_wins.load(Ordering::Relaxed),
            total_completed_nanos: self.total_completed_nanos.load(Ordering::Relaxed),
            max_completed_nanos: self.max_completed_nanos.load(Ordering::Relaxed),
        }
    }

    fn record_start(&self, length: usize, is_hedge: bool) {
        self.requests_started.fetch_add(1, Ordering::Relaxed);
        self.bytes_started
            .fetch_add(length as u64, Ordering::Relaxed);
        update_max(&self.max_bytes_started, length as u64);
        if is_hedge {
            self.hedge_requests_started.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_complete(&self, elapsed: Duration) {
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
        let nanos = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
        self.total_completed_nanos
            .fetch_add(nanos, Ordering::Relaxed);
        update_max(&self.max_completed_nanos, nanos);
    }

    fn record_hedge_win(&self) {
        self.hedge_wins.fetch_add(1, Ordering::Relaxed);
    }
}

/// An object store backed I/O source.
pub struct ObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: HostAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    hedge_config: Option<ObjectStoreReadHedgeConfig>,
    stats: Option<Arc<ObjectStoreReadStats>>,
}

impl ObjectStoreReadAt {
    /// Create a new object store source.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        Self::new_with_allocator(store, path, handle, Arc::new(DefaultHostAllocator))
    }

    /// Create a new object store source with a custom writable buffer allocator.
    pub fn new_with_allocator(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
            hedge_config: None,
            stats: None,
        }
    }

    /// Set the concurrency for this source.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the coalesce config for this source.
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.coalesce_config = Some(config);
        self
    }

    /// Set optional bounded request hedging for object-store range reads.
    pub fn with_hedge_config(mut self, config: ObjectStoreReadHedgeConfig) -> Self {
        self.hedge_config = Some(config);
        self
    }

    /// Attach physical object-store request counters.
    pub fn with_stats(mut self, stats: Arc<ObjectStoreReadStats>) -> Self {
        self.stats = Some(stats);
        self
    }
}

impl VortexReadAt for ObjectStoreReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        let hedge_config = self.hedge_config;
        let stats = self.stats.clone();
        let range = offset..(offset + length as u64);

        // Requires to deal with borrowed lifetimes
        let io_handle = handle.clone();

        handle
            .spawn_io(async move {
                let buffer = allocator.allocate(length, alignment)?;
                let buffer = read_range_with_optional_hedge(
                    store,
                    path,
                    io_handle,
                    buffer,
                    range,
                    length,
                    allocator,
                    alignment,
                    hedge_config,
                    stats,
                )
                .await?;

                Ok(BufferHandle::new_host(buffer.freeze()))
            })
            .boxed()
    }
}

async fn read_range_with_optional_hedge(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    buffer: WritableHostBuffer,
    range: std::ops::Range<u64>,
    length: usize,
    allocator: HostAllocatorRef,
    alignment: Alignment,
    hedge_config: Option<ObjectStoreReadHedgeConfig>,
    stats: Option<Arc<ObjectStoreReadStats>>,
) -> VortexResult<WritableHostBuffer> {
    if let Some(config) = hedge_config
        && config.applies_to(length)
    {
        #[cfg(not(target_arch = "wasm32"))]
        {
            return read_range_hedged(
                store, path, io_handle, buffer, range, length, allocator, alignment, config, stats,
            )
            .await;
        }
    }

    read_object_range(store, path, io_handle, buffer, range, length, false, stats).await
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_range_hedged(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    buffer: WritableHostBuffer,
    range: std::ops::Range<u64>,
    length: usize,
    allocator: HostAllocatorRef,
    alignment: Alignment,
    hedge_config: ObjectStoreReadHedgeConfig,
    stats: Option<Arc<ObjectStoreReadStats>>,
) -> VortexResult<WritableHostBuffer> {
    let primary = read_object_range(
        Arc::clone(&store),
        path.clone(),
        io_handle.clone(),
        buffer,
        range.clone(),
        length,
        false,
        stats.clone(),
    );
    futures::pin_mut!(primary);

    let delay = smol::Timer::after(hedge_config.delay);
    futures::pin_mut!(delay);

    match futures::future::select(primary, delay).await {
        Either::Left((result, _)) => result,
        Either::Right((_, primary)) => {
            let hedge_buffer = allocator.allocate(length, alignment)?;
            let hedge = read_object_range(
                store,
                path,
                io_handle,
                hedge_buffer,
                range,
                length,
                true,
                stats.clone(),
            );
            futures::pin_mut!(hedge);
            match futures::future::select(primary, hedge).await {
                Either::Left((result, _)) => result,
                Either::Right((result, _)) => {
                    if let Some(stats) = stats {
                        stats.record_hedge_win();
                    }
                    result
                }
            }
        }
    }
}

async fn read_object_range(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    mut buffer: WritableHostBuffer,
    range: std::ops::Range<u64>,
    length: usize,
    is_hedge: bool,
    stats: Option<Arc<ObjectStoreReadStats>>,
) -> VortexResult<WritableHostBuffer> {
    if let Some(stats) = &stats {
        stats.record_start(length, is_hedge);
    }
    let started = Instant::now();
    let response = store
        .get_opts(
            &path,
            GetOptions {
                range: Some(GetRange::Bounded(range.clone())),
                ..Default::default()
            },
        )
        .await?;

    let buffer = match response.payload {
        #[cfg(not(target_arch = "wasm32"))]
        GetResultPayload::File(file, _) => io_handle
            .spawn_blocking(move || {
                read_exact_at(&file, buffer.as_mut_slice(), range.start)?;
                Ok::<_, io::Error>(buffer)
            })
            .await
            .map_err(io::Error::other)?,
        #[cfg(target_arch = "wasm32")]
        GetResultPayload::File(..) => {
            unreachable!("File payload not supported on wasm32")
        }
        GetResultPayload::Stream(mut byte_stream) => {
            let mut written = 0usize;
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                let end = written + bytes.len();
                vortex_ensure!(
                    end <= length,
                    "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                    end,
                    length,
                    range
                );
                buffer.as_mut_slice()[written..end].copy_from_slice(&bytes);
                written = end;
            }

            vortex_ensure!(
                written == length,
                "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                written,
                length,
                range
            );

            buffer
        }
    };

    if let Some(stats) = &stats {
        stats.record_complete(started.elapsed());
    }
    Ok(buffer)
}

fn update_max(target: &AtomicU64, value: u64) {
    let mut current = target.load(Ordering::Relaxed);
    while value > current {
        match target.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }
}

#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use object_store::PutPayload;
    use object_store::memory::InMemory;

    use super::*;
    use crate::runtime::AbortHandle;
    use crate::runtime::AbortHandleRef;
    use crate::runtime::Executor;

    const TEST_DATA: &[u8] = b"object store test data";

    #[derive(Default)]
    struct CountingExecutor {
        spawn_count: AtomicUsize,
        spawn_io_count: AtomicUsize,
    }

    impl Executor for CountingExecutor {
        fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_io(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_io_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::spawn(async move { task() }).abort_handle())
        }

        fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::task::spawn_blocking(task).abort_handle())
        }
    }

    struct TokioAbortHandle(tokio::task::AbortHandle);

    impl TokioAbortHandle {
        fn new_handle(handle: tokio::task::AbortHandle) -> AbortHandleRef {
            Box::new(Self(handle))
        }
    }

    impl AbortHandle for TokioAbortHandle {
        fn abort(self: Box<Self>) {
            self.0.abort();
        }
    }

    #[tokio::test]
    async fn read_at_uses_spawn_io() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let stats = Arc::new(ObjectStoreReadStats::default());
        let reader = ObjectStoreReadAt::new(store, path, handle).with_stats(Arc::clone(&stats));
        let buffer = reader.read_at(7, 5, Alignment::new(1)).await?;

        assert_eq!(buffer.to_host().await.as_slice(), b"store");
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);
        let stats = stats.snapshot();
        assert_eq!(stats.requests_started, 1);
        assert_eq!(stats.requests_completed, 1);
        assert_eq!(stats.bytes_started, 5);
        assert_eq!(stats.max_bytes_started, 5);
        assert_eq!(stats.hedge_requests_started, 0);
        assert_eq!(stats.hedge_wins, 0);
        assert!(stats.total_completed_nanos > 0);
        assert!(stats.max_completed_nanos > 0);

        Ok(())
    }
}
