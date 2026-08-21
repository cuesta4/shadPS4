// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <chrono>
#include <filesystem>
#include <limits>

#include "common/logging/log.h"
#include "common/types.h"

namespace Common::PerformanceTelemetry {

using EventSeq = u64;
using FrameSeq = u64;
using CmdBufferSeq = u64;
using PacketSeq = u64;
using ProducerSeq = u64;
using FenceSeq = u64;
using FenceGen = u64;
using WaitSeq = u64;
using ResourceSeq = u64;
using ResourceVersion = u64;
using SubmitSeq = u64;
using CpuAccessSeq = u64;
using ReadbackSeq = u64;
using WatchSeq = u64;
using ResourceId = ResourceSeq;
using ResourceVer = ResourceVersion;

enum class CorrelationStatus : u8 {
    Complete,
    Partial,
    MissingContext,
    ContextEvicted,
    NotApplicable,
    Unsupported,
    InternalError,
    Count,
};

enum class CandidateRemoveReason : u8 {
    None,
    MatchedAndConsumed,
    SupersededByNewFence,
    ExplicitWriteToLabel,
    CpuWriteToLabel,
    Pm4WriteData,
    Pm4Dma,
    QueueReset,
    CommandBufferEnd,
    SubmissionBoundary,
    FrameBoundary,
    CapacityEviction,
    Invalidated,
    Unknown,
    Count,
};

enum class ConsumerProbeKind : u8 {
    MatchedFence,
    StructuralShadowFence,
    Count,
};

enum class ConsumerAccessPath : u8 {
    Unknown,
    SampledImage,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
    TexelBuffer,
    VertexBuffer,
    IndexBuffer,
    DrawIndirectArgs,
    DrawIndirectCount,
    DispatchIndirectArgs,
    ColorAttachment,
    DepthAttachment,
    ImageAliasResolve,
    BufferFromImage,
    GuestRamUpload,
    CpuFaultRead,
    Count,
};

enum class ConsumerConfidence : u8 {
    ExactResourceAndVersion,
    ExactResource,
    ExactGuestRange,
    PartialGuestOverlap,
    AliasBaseMatch,
    Heuristic,
    Count,
};

enum class MemoryWriteOrigin : u8 {
    GuestCpu,
    FenceSignal,
    ReadbackCommit,
    Pm4WriteData,
    Pm4Dma,
    VideoOut,
    EmulatorInternal,
    Unknown,
    Count,
};
using GuestMemoryWriteOrigin = MemoryWriteOrigin;

enum class ShadowBasis : u8 {
    None,
    AdjacentSameLabel,
    QueueOrderSameLabel,
    Heuristic,
    Count,
};

enum class ShadowOwner : u8 {
    Unknown,
    GuestRam,
    Buffer,
    Image,
    Cpu,
    Count,
};

enum class PromotionReason : u8 {
    FenceReadbackSource,
    FenceFirstConsumerSource,
    CpuObservation,
    AliasObservation,
    StaleGuestAttempt,
    DiagnosticTrigger,
    Count,
};

enum class FenceMatchFailure : u8 {
    None,
    NoCandidate,
    NoActiveGeneration,
    AddressMismatch,
    ReferenceMismatch,
    MaskMismatch,
    FunctionMismatch,
    GenerationSuperseded,
    CandidateAlreadyConsumed,
    QueueRejected,
    StageRejected,
    UnsupportedComparison,
    UnsupportedFenceKind,
    StateMissing,
    InternalCollision,
    Count,
};

enum class RepresentationTransition : u8 {
    ImageToImage,
    ImageToBuffer,
    ImageToIndirect,
    ImageToCpu,
    BufferToImage,
    BufferToBuffer,
    SameResource,
    Unknown,
    NotObserved,
    Count,
};

enum class TraceGapReason : u8 {
    RingOverwrite,
    Sampling,
    Filter,
    WriterBackpressure,
    CaptureDisabled,
    SchemaUnsupported,
    Count,
};

enum class TraceCaptureProfile : u8 {
    SyncPerf,
    SyncSemantic,
    SyncFastpathValidation,
    Count,
};

enum class FastpathEligibility : u8 {
    Eligible,
    Rejected,
    Count,
};

enum class FastpathRejectReason : u8 {
    None,
    TitleMismatch,
    ProducerNotCompute,
    ResourceNotStorage,
    SizeNot3072,
    TiledUnsupported,
    EosInvalidCommand,
    EosInvalidValue,
    EosHasIrq,
    EosSizeMismatch,
    WaitMismatch,
    AcquireMismatch,
    QueueOrderMismatch,
    LifetimeInvalid,
    OverlapConflict,
    Count,
};

enum class VirtualWaitResult : u8 {
    Virtualized,
    PhysicalWait,
    Fallback,
    Count,
};

enum class AsyncLabelAction : u8 {
    Wrote,
    StaleGenerationSkipped,
    AlreadySatisfied,
    Count,
};

enum class LazyMaterializeResult : u8 {
    Success,
    ImageGenerationMismatch,
    Unmapped,
    DownloadFailed,
    Fallback,
    Count,
};

enum class FastpathFallbackPhase : u8 {
    Recognizer,
    PendingDownload,
    Fence,
    Wait,
    Barrier,
    RamMaterialize,
    CpuFault,
    Lifetime,
    Overlap,
    Count,
};

enum class FastpathFallbackReason : u8 {
    None,
    SignatureMismatch,
    ResourceInvalid,
    QueueInvalid,
    OverlapConflict,
    LifetimeMismatch,
    BarrierFailed,
    DownloadFailed,
    UnknownOrigin,
    Count,
};

enum class ConservativeDownloadDecision : u8 {
    LegacyDownload,
    SuppressAuthority,
    MaterializeRequired,
    GenerationMismatchFallback,
    Count,
};

enum class FastpathWaitDecision : u8 {
    Virtualized,
    Legacy,
    ForcedComplete,
    Count,
};

enum class VirtualFenceForcedCompletionReason : u8 {
    HostSideEffect,
    RamDemand,
    CpuLabelVisibility,
    Irq,
    QueueTransition,
    Unmap,
    Fallback,
    Count,
};

enum class SourceWatchState : u8 {
    Active,
    GpuRead,
    GpuAliasMaterialization,
    GuestRamGpuUpload,
    CpuRead,
    Overwritten,
    DestroyedOrUnmapped,
    SessionEndUnknown,
    Count,
};
using TerminalKind = SourceWatchState;

enum class HostVersionOrigin : u8 {
    GuestCpu,
    ReadbackCommit,
    Pm4WriteData,
    Pm4Dma,
    FenceSignal,
    Unknown,
    Count,
};

enum class GuestSourceConsumePath : u8 {
    StreamBufferCopy,
    StagingBufferCopy,
    BufferUpload,
    ImageUpload,
    Count,
};

enum class ResourceLineageKind : u8 {
    GpuToGpu,
    CpuToGpu,
    Count,
};

enum class ReadWatchKind : u8 {
    Label,
    Data,
    Count,
};

enum class SemanticReadOrigin : u8 {
    None,
    GuestDirect,
    GuestHle,
    WaitRegMemPoll,
    GpuUploadFromGuestRam,
    RendererInternal,
    MemoryTrackerInternal,
    TelemetryInternal,
    UnknownHost,
    Count,
};

struct SemanticReadContext {
    SemanticReadOrigin origin{SemanticReadOrigin::None};
    u64 fence_seq{0};
    u64 readback_seq{0};
    u64 resource_id{0};
    u64 resource_version{0};
};

inline thread_local SemanticReadContext t_semantic_read_context{};

class ScopedSemanticReadOrigin {
public:
    explicit ScopedSemanticReadOrigin(SemanticReadOrigin origin, u64 fence_seq = 0,
                                     u64 readback_seq = 0, u64 res_id = 0, u64 res_ver = 0) noexcept
        : previous(t_semantic_read_context) {
        t_semantic_read_context = {
            .origin = origin,
            .fence_seq = fence_seq != 0 ? fence_seq : previous.fence_seq,
            .readback_seq = readback_seq != 0 ? readback_seq : previous.readback_seq,
            .resource_id = res_id != 0 ? res_id : previous.resource_id,
            .resource_version = res_ver != 0 ? res_ver : previous.resource_version,
        };
    }

    ~ScopedSemanticReadOrigin() noexcept {
        t_semantic_read_context = previous;
    }

    ScopedSemanticReadOrigin(const ScopedSemanticReadOrigin&) = delete;
    ScopedSemanticReadOrigin& operator=(const ScopedSemanticReadOrigin&) = delete;

private:
    SemanticReadContext previous;
};

inline thread_local bool t_is_guest_execution_thread{false};

class ScopedGuestExecutionThread {
public:
    explicit ScopedGuestExecutionThread(bool is_guest = true) noexcept
        : previous(t_is_guest_execution_thread) {
        t_is_guest_execution_thread = is_guest;
    }
    ~ScopedGuestExecutionThread() noexcept {
        t_is_guest_execution_thread = previous;
    }

    ScopedGuestExecutionThread(const ScopedGuestExecutionThread&) = delete;
    ScopedGuestExecutionThread& operator=(const ScopedGuestExecutionThread&) = delete;

private:
    bool previous;
};

struct GuestExecutableRange {
    VAddr begin{};
    VAddr end{};
};

class GuestExecutableRegistry {
public:
    static void RegisterRange(VAddr begin, VAddr end) noexcept {
        if (begin >= end) return;
        std::scoped_lock lock{s_mutex};
        const size_t count = s_count.load(std::memory_order_relaxed);
        if (count < MaxRanges) {
            s_ranges[count] = {begin, end};
            s_count.store(count + 1, std::memory_order_release);
        }
    }

    [[nodiscard]] static bool IsGuestRip(VAddr rip) noexcept {
        if (rip == 0) return false;
        const size_t count = s_count.load(std::memory_order_acquire);
        for (size_t i = 0; i < count; ++i) {
            if (rip >= s_ranges[i].begin && rip < s_ranges[i].end) {
                return true;
            }
        }
        return false;
    }

private:
    static constexpr size_t MaxRanges = 256;
    inline static std::array<GuestExecutableRange, MaxRanges> s_ranges{};
    inline static std::atomic<size_t> s_count{0};
    inline static std::mutex s_mutex;
};

enum class Counter : u16 {
    Pm4Packets,
    Pm4Type2Packets,
    DcbBytes,
    CcbBytes,
    AcbBytes,
    GfxSubmits,
    AscSubmits,
    GcpWakes,
    QueueScans,
    QueueResumes,
    QueueFrontLoads,
    GcpActiveNs,
    GcpBlockedNs,
    QueueReadyNs,
    QueueResumeNs,
    IbDepthMax,
    Draws,
    Dispatches,
    DrawCpuNs,
    DispatchCpuNs,
    PipelineHits,
    PipelineMisses,
    PipelineCompileNs,
    RenderTargetHits,
    RenderTargetMisses,
    ImageTokenHits,
    ImageTokenMisses,
    DescriptorHits,
    DescriptorMisses,
    BufferTokenHits,
    BufferTokenMisses,
    StreamSliceHits,
    StreamSliceMisses,
    StagingBytes,
    BarrierCalls,
    CopyCalls,
    CopyBytes,
    TimelinePolls,
    TimelinePollNs,
    PendingOpEmptyHits,
    PendingOpKnownTickHits,
    PendingOpRefreshes,
    StageCacheCurrentHits,
    StageCacheSearchHits,
    StageCacheMisses,
    StageCacheUncacheable,
    StageFingerprintCollisions,
    StageSpecializationBuilds,
    StageProgramCreates,
    StagePermutationCompiles,
    StagePermutationHits,
    FetchShaderCacheHits,
    FetchShaderCacheMisses,
    FetchShaderWords,
    DynamicStateHits,
    DynamicStateMisses,
    DynamicStateEmptyCommits,
    DriverSubmitCalls,
    DriverSubmitNs,
    DriverPresentCalls,
    DriverPresentNs,
    SubmitQueueDepthMax,
    WaitCalls,
    WaitNs,
    WritebackCalls,
    WritebackBytes,
    WritebackNs,
    WritebackEnqueueNs,
    WritebackBatches,
    WritebackStaleSkips,
    WritebackFenceDeferrals,
    WritebackFlushes,
    PresentPrepareNs,
    PresentCpuNs,
    GpuIdleGaps,
    GpuIdleGapNs,
    MemoryWatchArms,
    MemoryWatchCancels,
    MemoryWatchWakeups,
    MemoryWatchArmFailures,
    MemoryWatchFallbacks,
    MemoryNotifyCalls,
    MemoryNotifyPages,
    MemoryNotifyTrackedPages,
    MemoryNotifyCallbacks,
    MemoryNotifyNs,
    MemoryNotifyCpu,
    MemoryNotifyCommandProcessor,
    MemoryNotifyGpuCompletion,
    MemoryNotifyMap,
    MemoryNotifyUnmap,
    DynamicStateInvalidations,
    RenderColorAttachments,
    RenderDepthAttachments,
    RenderTargetRebinds,
    SubmitMutexWaitNs,
    SubmitMutexHoldNs,
    SubmitPrepareNs,
    SubmitPostNs,
    PresentMutexWaitNs,
    PresentMutexHoldNs,
    MemoryNotifyNoWatchCalls,
    MemoryNotifyActiveWatchCalls,
    EventQueryCalls,
    EventQueryCounterPairs,
    EventQueryNoWatchCalls,
    EventQueryActiveWatchCalls,
    EventQueryMatchedWatchCalls,
    WritebackDrainCalls,
    WritebackDrainEmpty,
    WritebackCandidates,
    WritebackSameEpoch,
    StageSlowCurrentHits,
    StageSlowOtherHits,
    StageSlowSearchComparisons,
    StagingBatchSamples,
    StagingBatchRequests,
    StagingBatchCanonicalCopies,
    StagingBatchGuestCopies,
    StagingBatchExactReuses,
    StagingBatchSubrangeReuses,
    StagingBatchRequestedBytes,
    StagingBatchCanonicalBytes,
    StagingBatchAllocatedBytes,
    StagingSparseCopySamples,
    StagingSparseCopyRequests,
    StagingSparsePlanHits,
    StagingSparsePlanMisses,
    StagingSparsePlansBuilt,
    StagingSparseCopyReferenceFallbacks,
    StagingSparseMappedRuns,
    StagingSparseZeroRuns,
    StagingSparseRequestedBytes,
    StagingSparseCopiedBytes,
    StagingSparseReferenceBytes,
    StagingSparseMappedBytes,
    StagingSparseZeroBytes,
    StagingSparsePlanMissEmpty,
    StagingSparsePlanMissGeneration,
    StagingSparsePlanMissConflict,
    StagingSparsePlanMissUncacheable,
    StagingSparseNonTemporalRuns,
    StagingSparseNonTemporalBytes,
    StagingSparseNonTemporalFences,
    StagingSparseMergeablePairs,
    StagingSparseMergeableBytes,
    DescriptorCrossPipelineSamples,
    DescriptorCrossPipelineExactStateHits,
    DescriptorCrossPipelineCompatibleLayoutHits,
    DescriptorCrossPipelineReusableHits,
    StagingSparseDenseHotHits,
    StagingSparseDenseCacheHits,
    StagingSparseDenseRefills,
    StagingSparseDenseFallbackRequests,
    StagingSparseDenseHotBytes,
    StagingSparseDenseCacheBytes,
    StagingSparseDenseFallbackBytes,
    StagingCurrentStreamBatches,
    StagingCurrentStreamBytes,
    StagingHostDirectBatches,
    StagingHostDirectBytes,
    StagingGpuPromotedBatches,
    StagingGpuPromotedBytes,
    StagingReuseCandidates,
    StagingReuseHits,
    StagingReuseMismatches,
    StagingReuseCooldownSkips,
    StagingReuseInvalidations,
    StagingReuseWarmups,
    StagingReuseCandidateBytes,
    StagingReuseAvoidedBytes,
    StagingReuseWarmupBytes,
    WritebackFlushesAvoided,
    GpuFenceWaitBypasses,
    GpuFenceWaitBarriers,
    GpuFenceWaitForcedFlushes,
    GpuFenceTokenOverflows,
    AcquireMemCalls,
    AcquireMemBarriers,
    EventWriteFlushCalls,
    EventWriteFlushBarriers,
    RenderTargetSyncBarriers,
    RenderTargetSampledAsTexture,
    RenderTargetTransitions,
    BruteForceBarriers,
    BruteForceHostFinishes,
    WaitRegMemCalls,
    WaitRegMemSpinNs,
    WaitRegMemSpins,
    PriorityOpsWaitNs,
    PriorityOpsDrainCount,
    PriorityOpsExecuteNs,
    FenceTotalCount,
    FenceGpuOnlyCount,
    FenceCpuVisibleCount,
    FenceAmbiguousCount,
    FenceEpochLinks,
    WaitMatchedFences,
    WaitUnmatchedFences,
    CpuAccessCount,
    CpuStaleAccessCount,
    CpuMaterializationCount,
    StaleGuestAttempts,
    GpuAliasMaterializations,
    ReadbackScheduleCount,
    ReadbackSubmitCount,
    ReadbackReadyCount,
    ReadbackCommitCount,
    FenceSignalCount,
    HostWaitCount,
    HostWaitFinishCount,
    HostWaitCpuVisibilityCount,
    HostWaitWaitProgressCount,
    FenceMatchAttemptCount,
    FenceMatchSuccessCount,
    FenceMatchFailureCount,
    WaitCompleteCount,
    FirstConsumerCount,
    CpuLabelAccessCount,
    TraceGapCount,
    RingHealthCount,
    FenceMatchDiagnosticCount,
    ResourceEpochPromotedCount,
    FenceResourceLinkCount,
    ShadowConsumerProbeHits,
    MatchedConsumerProbeHits,
    SemanticLabelWatchArms,
    SemanticDataWatchArms,
    SemanticLabelWatchPages,
    SemanticDataWatchPages,
    SemanticReadFaults,
    CpuReadObservations,
    SemanticReadFaultsTotal,
    SemanticReadGuestDirect,
    SemanticReadGuestHle,
    SemanticReadWaitRegMemPoll,
    SemanticReadGpuUpload,
    SemanticReadRendererInternal,
    SemanticReadMemoryTrackerInternal,
    SemanticReadTelemetryInternal,
    SemanticReadUnknownHost,
    SemanticLabelGuestReads,
    SemanticDataGuestReads,
    SemanticPrecisionUnknownReads,
    SemanticActiveLabelWatches,
    SemanticActiveDataWatches,
    SemanticPageConflictWrites,
    SemanticWatchCancelWaitComplete,
    SemanticWatchCancelOverwritten,
    SemanticWatchCancelPageConflict,
    SemanticFaultLivelockBreaks,
    FastpathCandidates,
    FastpathTaken,
    FastpathRejectedSignature,
    FastpathRejectedResource,
    FastpathRejectedQueue,
    FastpathRejectedOverlap,
    FastpathRejectedLifetime,
    Eager3kDownloads,
    Lazy3kMaterializations,
    AuthorityCreated,
    AuthorityGpuFirstConsumer,
    AuthorityRamFirstConsumer,
    AuthorityCpuRead,
    AuthoritySupersededWithoutHostUse,
    AuthorityDestroyedWithoutHostUse,
    VirtualWaitConsumed,
    PhysicalWaitFallback,
    AsyncLabelWrites,
    StaleLabelCallbacks,
    BarrierValidationSuccess,
    BarrierValidationFailure,
    RamDemandEvents,
    RamDemandUniqueAuthorities,
    MaterializeSuccess,
    MaterializeFailure,
    MaterializeByteMismatch,
    ConservativeDownloadConsidered,
    ConservativeDownloadSuppressed,
    Conservative3kReadbacks,
    FastpathWaitProgressSubmits,
    VirtualFenceForcedCompletions,
    VirtualWaitFallback,
    Count,
};

enum class EventType : u16 {
    None,
    GcpActive,
    GcpBlocked,
    SubmitDone,
    TimelineComplete,
    DriverSubmit,
    Wait,
    Writeback,
    PresentPrepare,
    PresentCpu,
    DriverPresent,
    FramePresented,
    GpuIdleGap,
    Pm4Packet,
    Pm4WaitBegin,
    Pm4WaitEnd,
    Pm4AcquireMem,
    Pm4ReleaseMem,
    Pm4EventWrite,
    Pm4SurfaceSync,
    VulkanPipelineBarrier,
    VulkanImageLayout,
    VulkanDraw,
    VulkanDispatch,
    VulkanCopy,
    VulkanSubmit,
    MemoryWatchArm,
    MemoryWatchWake,
    AliasSync,
    SyncPm4Packet,
    ProducerBegin,
    ProducerEnd,
    ResourceWrite,
    FenceCreate,
    FenceEpochLink,
    WaitCreate,
    FenceClassification,
    CpuMemoryAccess,
    CpuReadRequiresMaterialization,
    StaleGuestSourceAttempt,
    GpuAliasMaterialize,
    ReadbackSchedule,
    ReadbackSubmit,
    ReadbackReady,
    ReadbackCommit,
    FenceSignal,
    HostWait,
    SubmitRecord,
    FenceMatchAttempt,
    WaitComplete,
    FirstConsumer,
    CpuLabelAccess,
    TraceGap,
    RingHealth,
    ProducerRecord,
    ShadowFencePolicy,
    FenceMatchDiagnostic,
    ResourceEpochPromoted,
    FenceResourceLink,
    Count,
};

enum class Pm4Engine : u8 {
    Constant,
    Graphics,
    Compute,
};

enum class ProducerClass : u8 {
    GraphicsDraw,
    ComputeDispatch,
    ComputeHle,
    Clear,
    Copy,
    Resolve,
    ColorTarget,
    DepthTarget,
    StorageImage,
    StorageBuffer,
    Dma,
    Cpu,
    Count,
};

enum class ResourceType : u8 {
    Image,
    Buffer,
    ColorTarget,
    DepthTarget,
    DownloadBuffer,
    Count,
};

enum class ResourceWriteKind : u8 {
    StorageImage,
    StorageBuffer,
    ColorTarget,
    DepthTarget,
    Transfer,
    Resolve,
    Clear,
    Dma,
    Cpu,
    Count,
};

enum class FenceKind : u8 {
    EventWriteEos,
    EventWriteEop,
    ReleaseMem,
    WriteData,
    Count,
};

enum class FenceStageScope : u8 {
    Cs,
    Ps,
    Pipe,
    All,
    Count,
};

enum class FenceClassification : u8 {
    GpuOnly,
    CpuVisible,
    Ambiguous,
    Count,
};

enum class FenceEvidence : u32 {
    None = 0,
    MatchedGpuWait = 1 << 0,
    CpuReadObserved = 1 << 1,
    CpuWriteObserved = 1 << 2,
    InterruptRequested = 1 << 3,
    MultipleConsumers = 1 << 4,
    UnknownCommandBuffer = 1 << 5,
    CrossQueue = 1 << 6,
    GuestDataReadObserved = 1 << 7,
};

enum class FenceLinkReason : u8 {
    CandidateWriteEpoch,
    DirectEpochRef,
    OverlapHeuristic,
    StructuralShadow,
    SameQueuePriorProducer,
    SameStagePriorProducer,
    ExplicitCacheScope,
    GlobalConservative,
    Count,
};

enum class WaitConfidence : u8 {
    None,
    Low,
    Medium,
    High,
    Exact,
    Count,
};

enum class CpuAccessType : u8 {
    Read,
    Write,
    Count,
};

enum class FallbackAction : u8 {
    Allowed,
    ForcedHostReadback,
    GpuImageToBuffer,
    GpuImageToImage,
    GpuBufferToImage,
    Unknown,
    Count,
};

enum class AliasCopyKind : u8 {
    ImageToImage,
    ImageToBuffer,
    BufferToImage,
    Resolve,
    Retile,
    FormatReinterpret,
    ComputeConversion,
    Count,
};

enum class ReadbackReason : u8 {
    FenceConservative,
    CpuReadFault,
    PreemptiveHotRange,
    AliasFallback,
    Explicit1x1,
    DebugValidation,
    Count,
};

enum class HostWaitReason : u8 {
    Unknown,
    SchedulerFinish,
    FenceCpuVisibility,
    WaitRegMemProgress,
    StreamBufferReuse,
    ResourceDestruction,
    Present,
    Debug,
    Count,
};

enum class TimerSite : u8 {
    StageRefresh,
    StageFlatCopy,
    StageSrtWalker,
    StageResolve,
    StageCurrentMatch,
    StageSearch,
    StageSpecializationBuild,
    StageSlowPath,
    DynamicTotal,
    DynamicViewport,
    DynamicDepthStencil,
    DynamicPrimitive,
    DynamicRasterization,
    DynamicBlend,
    DescriptorPrepare,
    DescriptorCompare,
    DescriptorEmit,
    RenderPrepare,
    RenderImageDesc,
    RenderStateBuild,
    ImageFind,
    ImageUpdate,
    RenderTargetPrepare,
    SchedulerBeginRendering,
    MemoryNotify,
    EventQueryStores,
    EventQueryNotify,
    StagingStreamSingle,
    StagingStreamBatch,
    StagingStreamSlice,
    StagingImage,
    StagingUploadCopies,
    StagingWriteData,
    StagingBatchDeduplicate,
    StagingBatchLayout,
    StagingBatchCopy,
    StagingBatchResults,
    StagingBatchRequestPrepare,
    StagingSparseCopy,
    StagingSparseSharedTotal,
    StagingSparseLock,
    StagingSparsePlanLookup,
    StagingSparsePlanBuild,
    StagingSparsePayloadCopy,
    StagingSparseFinish,
    StagingSparseDenseLookup,
    StagingSparseSpanRefill,
    StagingSparseColdPath,
    StagingStreamReuse,
    DescriptorUserData,
    DescriptorBuffers,
    DescriptorTextures,
    DescriptorCapture,
    Count,
};

enum class StageUncacheableReason : u32 {
    SrtWalker = 1U << 0,
    TessellationControl = 1U << 1,
    TessellationEvaluation = 1U << 2,
    DescriptorOutsideUserData = 1U << 3,
    FetchPointerOutsideUserData = 1U << 4,
    FetchShaderUnavailable = 1U << 5,
};

enum class ImageFindPath : u8 {
    ExactCache,
    PerfectScan,
    Overlap,
    Created,
    Count,
};

enum class StagingSite : u8 {
    StreamSingle,
    StreamBatch,
    StreamSlice,
    Image,
    UploadCopies,
    WriteData,
    Count,
};

enum class StagingSource : u8 {
    Guest,
    Host,
    Zero,
    Count,
};

enum class StagingBackend : u8 {
    CurrentStream,
    HostDirect,
    GpuPromoted,
    Count,
};

enum class StagingMemoryKind : u8 {
    CurrentStream,
    HostDirect,
    Count,
};

struct StagingBatchSample {
    u32 requests{};
    u32 canonical_copies{};
    u32 guest_copies{};
    u32 host_copies{};
    u32 zero_copies{};
    u32 exact_reuses{};
    u32 subrange_reuses{};
    u64 requested_bytes{};
    u64 canonical_bytes{};
    u64 allocated_bytes{};
    u64 guest_bytes{};
    u64 host_bytes{};
    u64 zero_bytes{};
    u32 reuse_candidates{};
    u32 reuse_hits{};
    u32 reuse_mismatches{};
    u32 reuse_cooldown_skips{};
    u32 reuse_invalidations{};
    u32 reuse_warmups{};
    u64 reuse_candidate_bytes{};
    u64 reuse_avoided_bytes{};
    u64 reuse_warmup_bytes{};
};

struct StagingSparseCopySample {
    u32 requests{};
    u32 plan_hits{};
    u32 plan_misses{};
    u32 plans_built{};
    u32 copy_reference_fallbacks{};
    u32 mapped_runs{};
    u32 zero_runs{};
    u32 plan_miss_empty{};
    u32 plan_miss_generation{};
    u32 plan_miss_conflict{};
    u32 plan_miss_uncacheable{};
    u32 non_temporal_runs{};
    u32 non_temporal_fences{};
    u32 mergeable_pairs{};
    u32 dense_hot_hits{};
    u32 dense_cache_hits{};
    u32 dense_refills{};
    u32 dense_fallback_requests{};
    u64 requested_bytes{};
    u64 copied_bytes{};
    u64 reference_bytes{};
    u64 mapped_bytes{};
    u64 zero_bytes{};
    u64 non_temporal_bytes{};
    u64 mergeable_bytes{};
    u64 dense_hot_bytes{};
    u64 dense_cache_bytes{};
    u64 dense_fallback_bytes{};
};

enum class WritebackTrigger : u8 {
    GuestSubmit,
    EventWriteEos,
    EventWriteEop,
    ReleaseMem,
    GarbageCollection,
    Count,
};

enum class ImageWriter : u8 {
    None,
    GraphicsDraw,
    ComputeDispatch,
    ComputeHle,
    Transfer,
    CpuUpload,
    Count,
};

enum class SubmitReason : u8 {
    Generic,
    GuestSubmit,
    WritebackEos,
    WritebackEop,
    WritebackReleaseMem,
    PresentFrameBuild,
    PresentSubmit,
    QueuePresent,
    Finish,
    WaitProgress,
    Count,
};

struct WritebackImageSample {
    WritebackTrigger trigger{};
    ImageWriter writer{};
    u32 trigger_control{};
    u32 trigger_data_control{};
    u32 image_index{};
    u32 queued_images{};
    u64 image_uid{};
    u64 backing_image{};
    VAddr guest_address{};
    u64 download_bytes{};
    u64 content_epoch{};
    u64 previous_epoch{};
    u64 previous_backing{};
    u32 flags{};
};

inline constexpr u32 WritebackFlagSameEpoch = 1U << 0;
inline constexpr u32 WritebackFlagSameBacking = 1U << 1;
inline constexpr u32 WritebackFlagTiled = 1U << 2;
inline constexpr u32 WritebackFlagNarrow = 1U << 3;
inline constexpr u32 WritebackFlagStorage = 1U << 4;
inline constexpr u32 WritebackFlagRenderTarget = 1U << 5;

inline constexpr u32 InvalidTimerStage = std::numeric_limits<u32>::max();

[[nodiscard]] constexpr u32 TimerSamplePeriod(TimerSite site) noexcept {
    switch (site) {
    case TimerSite::StagingStreamSingle:
    case TimerSite::StagingStreamBatch:
    case TimerSite::StagingStreamSlice:
    case TimerSite::StagingImage:
    case TimerSite::StagingUploadCopies:
    case TimerSite::StagingWriteData:
    case TimerSite::StagingBatchDeduplicate:
    case TimerSite::StagingBatchLayout:
    case TimerSite::StagingBatchCopy:
    case TimerSite::StagingBatchResults:
    case TimerSite::StagingBatchRequestPrepare:
    case TimerSite::StagingSparseCopy:
    case TimerSite::StagingSparseSharedTotal:
    case TimerSite::StagingSparseLock:
    case TimerSite::StagingSparsePlanLookup:
    case TimerSite::StagingSparsePlanBuild:
    case TimerSite::StagingSparsePayloadCopy:
    case TimerSite::StagingSparseFinish:
    case TimerSite::StagingSparseDenseLookup:
    case TimerSite::StagingSparseSpanRefill:
    case TimerSite::StagingSparseColdPath:
    case TimerSite::StagingStreamReuse:
        return 64;
    case TimerSite::DescriptorPrepare:
    case TimerSite::DescriptorCompare:
    case TimerSite::DescriptorEmit:
    case TimerSite::ImageFind:
    case TimerSite::ImageUpdate:
    case TimerSite::DescriptorUserData:
    case TimerSite::DescriptorBuffers:
    case TimerSite::DescriptorTextures:
    case TimerSite::DescriptorCapture:
        return 512;
    case TimerSite::MemoryNotify:
    case TimerSite::EventQueryStores:
    case TimerSite::EventQueryNotify:
        return 1024;
    default:
        return 256;
    }
}

[[nodiscard]] inline bool Enabled() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Log::IsEnabled();
#else
    return false;
#endif
}

class Gate {
public:
    constexpr Gate& operator=(bool value) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        enabled = value;
#else
        static_cast<void>(value);
#endif
        return *this;
    }

    [[nodiscard]] constexpr operator bool() const noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        return enabled;
#else
        return false;
#endif
    }

private:
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    bool enabled{};
#endif
};

[[nodiscard]] inline u64 Timestamp() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void AddEnabled(Counter counter, u64 value) noexcept;
void ObserveMaxEnabled(Counter counter, u64 value) noexcept;
void RecordEnabled(EventType type, u64 arg0, u64 arg1) noexcept;
void RecordDurationEnabled(Counter counter, EventType type, u64 start_ns, u64 arg0) noexcept;
void RecordDurationValueEnabled(Counter counter, EventType type, u64 duration, u64 arg0) noexcept;
void RecordTimerSampleEnabled(TimerSite site, u64 start_ns,
                              u32 stage = InvalidTimerStage) noexcept;
void RecordTimerSampleDurationEnabled(TimerSite site, u64 duration_ns,
                                      u32 stage = InvalidTimerStage) noexcept;
void RecordStageUncacheableEnabled(u32 stage, u32 reason_mask) noexcept;
void RecordStageSlowResultEnabled(bool hit, bool current, u64 comparisons) noexcept;
void RecordDynamicStateDecisionEnabled(u32 reason_mask) noexcept;
void RecordDynamicCommitEnabled(u32 pending_groups, u32 emitted_groups) noexcept;
void RecordDescriptorDecisionEnabled(u32 reason_mask) noexcept;
void RecordImageFindPathEnabled(ImageFindPath path) noexcept;
void RecordStagingAllocationEnabled(StagingSite site, u64 bytes) noexcept;
void RecordStagingSourceEnabled(StagingSite site, StagingSource source, u64 bytes) noexcept;
void RecordStagingBatchEnabled(const StagingBatchSample& sample, bool sampled) noexcept;
[[nodiscard]] bool ShouldSampleStagingBatchEnabled() noexcept;
void RecordStagingSparseCopyEnabled(const StagingSparseCopySample& sample) noexcept;
[[nodiscard]] bool ShouldSampleStagingSparsePhaseEnabled() noexcept;
void RecordStagingBackendEnabled(StagingBackend backend, u64 bytes) noexcept;
void RecordStagingMemoryTypeEnabled(StagingMemoryKind kind, u32 memory_type, u32 memory_heap,
                                    u32 property_flags) noexcept;
[[nodiscard]] bool ShouldSampleDescriptorCrossPipelineEnabled() noexcept;
void RecordDescriptorCrossPipelineEnabled(bool exact_state, bool compatible_layout) noexcept;
void RecordSubmitTimingEnabled(SubmitReason reason, u64 mutex_wait_ns, u64 prepare_ns,
                               u64 driver_ns, u64 post_ns, u64 mutex_hold_ns) noexcept;
void RecordPresentTimingEnabled(u64 mutex_wait_ns, u64 driver_ns, u64 mutex_hold_ns) noexcept;
void RecordWritebackDrainEnabled(WritebackTrigger trigger, u32 candidates,
                                 u32 scheduled) noexcept;
void RecordWritebackImageEnabled(const WritebackImageSample& sample) noexcept;
void RecordEventQueryEnabled(u32 counter_pairs, bool had_active_watches,
                             u32 matched_pages, u32 callbacks) noexcept;
void RecordFrameSampleEnabled(u32 frame_id, u64 present_start_ns) noexcept;
void CountPm4PacketEnabled(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                           uintptr_t address, u32 words, u32 header) noexcept;

struct FenceTraceToken {
    FenceSeq fence_seq{0};
    FenceGen generation{0};
    PacketSeq packet_seq{0};
    VAddr label_addr{0};
    u64 label_val{0};
};

struct ReadbackTraceToken {
    ReadbackSeq readback_seq{0};
    FenceSeq fence_seq{0};
    ProducerSeq producer_seq{0};
    ResourceSeq resource_id{0};
    ResourceVersion resource_version{0};
    u64 gpu_tick{0};
};

struct SubmitTraceToken {
    SubmitSeq submit_seq{0};
    u64 signal_tick{0};
};

struct PendingOpTraceToken {
    FenceSeq fence_seq{0};
    ReadbackSeq readback_seq{0};
    SubmitSeq submit_seq{0};
    PacketSeq packet_seq{0};
};

struct FenceCandidateDebugState {
    FenceSeq last_insert_fence{};
    FenceGen last_insert_generation{};
    PacketSeq last_insert_packet{};
    FenceSeq last_remove_fence{};
    PacketSeq last_remove_packet{};
    CandidateRemoveReason last_remove_reason{CandidateRemoveReason::None};
    FenceSeq last_superseded_fence{};
    PacketSeq last_supersede_packet{};
    u64 last_value{};
    u64 last_mask{};
    u32 insert_count{};
    u32 remove_count{};
    u32 supersede_count{};
};

struct FenceMatchDiagnosticSample {
    WaitSeq wait_seq{};
    PacketSeq wait_packet_seq{};
    VAddr wait_addr{};
    u32 wait_ref{};
    u32 wait_mask{};
    u32 wait_func{};
    FenceSeq shadow_fence_seq{};
    FenceSeq last_insert_fence{};
    FenceGen last_insert_generation{};
    PacketSeq last_insert_packet{};
    FenceSeq last_remove_fence{};
    PacketSeq last_remove_packet{};
    CandidateRemoveReason last_remove_reason{CandidateRemoveReason::None};
    FenceSeq last_supersede_fence{};
    PacketSeq last_supersede_packet{};
    bool current_candidate_map_contains_address{false};
    u32 candidate_map_size{0};
    u32 packets_since_last_insert{0};
    u32 packets_since_last_remove{0};
};

struct ResourceEpochPromotedSample {
    ResourceSeq resource_id{};
    ResourceVersion resource_version{};
    ProducerSeq producer_seq{};
    PacketSeq producer_packet_seq{};
    u64 producer_timestamp{};
    VAddr guest_addr{};
    u64 size{};
    ResourceWriteKind write_kind{ResourceWriteKind::StorageImage};
    PromotionReason promotion_reason{PromotionReason::FenceReadbackSource};
};

struct FenceResourceLinkSample {
    FenceSeq fence_seq{};
    ResourceSeq resource_id{};
    ResourceVersion resource_version{};
    ProducerSeq producer_seq{};
    VAddr guest_addr{};
    u64 size{};
    ResourceType resource_kind{ResourceType::Image};
    ResourceWriteKind write_kind{ResourceWriteKind::StorageImage};
    FenceLinkReason reason{FenceLinkReason::CandidateWriteEpoch};
    ConsumerConfidence confidence{ConsumerConfidence::ExactResourceAndVersion};
};

struct SyncPm4PacketSample {
    FrameSeq frame_seq{};
    CmdBufferSeq cmd_buffer_seq{};
    PacketSeq packet_seq{};
    u32 queue_id{};
    Pm4Engine engine{};
    u32 ib_depth{};
    u32 opcode{};
    uintptr_t packet_addr{};
    VAddr label_addr{};
    u64 label_value{};
    u32 event_type{};
    u32 event_index{};
    u32 interrupt_select{};
    u32 data_select{};
    u32 cache_action{};
};

struct ProducerBeginSample {
    ProducerSeq producer_seq{};
    PacketSeq packet_seq{};
    FrameSeq frame_seq{};
    ProducerClass producer_type{};
    u32 queue_id{};
    u32 stage{};
    u64 shader_hash{};
    u64 pipeline_hash{};
    u32 dispatch_x{};
    u32 dispatch_y{};
    u32 dispatch_z{};
    u32 draw_count{};
};

struct ProducerEndSample {
    ProducerSeq producer_seq{};
    u32 write_range_count{};
    u64 write_bytes{};
    u32 write_resource_count{};
};

struct ResourceWriteSample {
    ProducerSeq producer_seq{};
    ResourceSeq resource_id{};
    ResourceType resource_type{};
    VAddr guest_addr{};
    u64 guest_size{};
    ResourceVersion version{};
    u64 vk_handle_id{};
    u32 format{};
    u32 width{};
    u32 height{};
    u32 depth{};
    u32 pitch{};
    u32 tiling{};
    ResourceWriteKind write_kind{};
    u32 stage{};
    u32 queue_id{};
};

struct FenceCreateSample {
    FenceSeq fence_seq{};
    FenceGen generation{};
    PacketSeq packet_seq{};
    FrameSeq frame_seq{};
    FenceKind fence_kind{};
    u32 raw_event_type{};
    u32 decoded_event_type{};
    u32 event_index{};
    u32 command{};
    VAddr label_addr{};
    u64 label_value{};
    u32 label_size{};
    u32 queue_id{};
    FenceStageScope stage_scope{};
    u32 interrupt_select{};
    u32 interrupt_id{};
    u32 data_select{};
    ProducerSeq producer_begin_seq{};
    ProducerSeq producer_end_seq{};
    u32 candidate_write_epoch_count{};
    u64 candidate_write_bytes{};
    FenceClassification classification_initial{};
    u32 classification_reason_bits{};
    CorrelationStatus correlation_status{};
};

struct FenceEpochLinkSample {
    FenceSeq fence_seq{};
    ProducerSeq producer_seq{};
    ResourceSeq resource_id{};
    ResourceVersion version{};
    VAddr guest_addr{};
    u64 guest_size{};
    ResourceType resource_kind{};
    ResourceWriteKind write_kind{};
    u32 queue_id{};
    u32 stage{};
    FenceLinkReason reason{};
};

struct FenceMatchAttemptSample {
    WaitSeq wait_seq{};
    PacketSeq packet_seq{};
    FrameSeq frame_seq{};
    u32 queue_id{};
    Pm4Engine engine{};
    VAddr wait_addr{};
    u32 ref{};
    u32 mask{};
    u32 function{};
    PacketSeq previous_packet_seq{};
    u32 previous_opcode{};
    u32 candidate_count{};
    FenceSeq nearest_candidate_fence_seq{};
    FenceGen nearest_candidate_generation{};
    bool candidate_address_match{};
    bool candidate_value_match{};
    bool candidate_mask_match{};
    bool candidate_function_match{};
    u32 packet_distance{};
    bool result_matched{};
    FenceMatchFailure failure_reason{};
};

struct WaitCreateSample {
    WaitSeq wait_seq{};
    PacketSeq packet_seq{};
    FrameSeq frame_seq{};
    u32 queue_id{};
    Pm4Engine engine{};
    VAddr wait_addr{};
    u32 ref{};
    u32 mask{};
    u32 function{};
    FenceSeq matched_fence_seq{};
    FenceGen matched_generation{};
    WaitConfidence matched_fence_confidence{};
    u32 commands_since_fence{};
    u32 packets_since_fence{};
    u32 producer_ops_between{};
    FenceSeq shadow_fence_seq{};
    ShadowBasis shadow_basis{ShadowBasis::None};
    CorrelationStatus matcher_status{CorrelationStatus::Complete};
    FenceMatchFailure failure_reason{FenceMatchFailure::None};
};

struct WaitCompleteSample {
    WaitSeq wait_seq{};
    FenceSeq fence_seq{};
    u64 start_timestamp{};
    u64 end_timestamp{};
    u64 duration_ns{};
    u64 spin_iterations{};
    u64 yield_count{};
    u32 exit_reason{};
    u32 value_at_begin{};
    u32 value_at_end{};
    u64 gpu_completed_tick_begin{};
    u64 gpu_completed_tick_end{};
    FenceSeq shadow_fence_seq{};
};

struct FirstConsumerSample {
    FenceSeq fence_seq{};
    WaitSeq wait_seq{};
    ConsumerProbeKind probe_kind{ConsumerProbeKind::MatchedFence};
    ProducerSeq producer_seq_source{};
    ProducerSeq producer_seq_consumer{};
    PacketSeq consumer_packet_seq{};
    u32 packet_distance_from_wait{};
    ResourceSeq source_resource_id{};
    ResourceVersion source_version{};
    ResourceSeq consumer_resource_id{};
    ResourceVersion consumer_resource_version{};
    ProducerClass consumer_type{};
    VAddr guest_overlap_addr{};
    u64 guest_overlap_size{};
    ConsumerAccessPath access_path{ConsumerAccessPath::Unknown};
    RepresentationTransition transition{};
    ConsumerConfidence confidence{ConsumerConfidence::ExactResourceAndVersion};
    u64 consumer_pipeline_hash{};
    u64 consumer_shader_hash{};
};

struct FenceClassificationSample {
    FenceSeq fence_seq{};
    u32 gpu_wait_count{};
    u32 cpu_label_read_count{};
    u32 cpu_label_write_count{};
    bool irq_requested{};
    u32 cpu_protected_data_read_count{};
    FenceClassification classification_final{};
    WaitConfidence confidence{};
    u32 reason_bits{};
};

struct CpuAccessSample {
    CpuAccessSeq cpu_access_seq{};
    FrameSeq frame_seq{};
    u32 thread_id{};
    CpuAccessType access_type{};
    GuestMemoryWriteOrigin write_origin{};
    VAddr guest_addr{};
    u64 size{};
    bool is_label_range{};
    FenceSeq matched_fence_seq{};
    ResourceVersion latest_version{};
    ResourceVersion host_version{};
    ResourceType authoritative_owner{};
    ResourceSeq authoritative_resource{};
    u64 authoritative_tick{};
};

struct CpuLabelAccessSample {
    FenceSeq fence_seq{};
    FenceGen generation{};
    CpuAccessType access_type{};
    VAddr guest_addr{};
    u64 timestamp_ns{};
    u32 guest_thread_id{};
};

struct CpuMaterializationSample {
    CpuAccessSeq cpu_access_seq{};
    VAddr guest_addr{};
    u64 size{};
    ResourceVersion latest_version{};
    ResourceVersion host_version{};
    ProducerSeq producer_seq{};
    u64 producer_tick{};
    ResourceSeq source_resource_id{};
    ResourceType source_resource_type{};
    bool readback_already_scheduled{};
    bool readback_ready{};
};

struct StaleGuestAttemptSample {
    FrameSeq frame_seq{};
    PacketSeq packet_seq{};
    VAddr guest_addr{};
    u64 size{};
    ResourceSeq consumer_resource_id{};
    ResourceType consumer_type{};
    ResourceVersion host_version{};
    ResourceVersion latest_version{};
    ResourceSeq authoritative_resource_id{};
    ResourceType authoritative_resource_type{};
    FallbackAction fallback_action{};
};

struct GpuAliasMaterializeSample {
    ResourceSeq source_resource_id{};
    ResourceVersion source_version{};
    ResourceSeq dest_resource_id{};
    ResourceVersion dest_previous_version{};
    ResourceVersion dest_new_version{};
    VAddr guest_addr{};
    u64 size{};
    AliasCopyKind copy_kind{};
    SubmitSeq submit_seq{};
    PacketSeq packet_seq{};
};

struct ReadbackScheduleSample {
    ReadbackSeq readback_seq{};
    FenceSeq fence_seq{};
    ResourceSeq resource_id{};
    ResourceVersion version{};
    VAddr guest_addr{};
    u64 size{};
    u64 download_offset{};
    u64 producer_tick{};
    u64 schedule_tick{};
    ReadbackReason reason{};
    CorrelationStatus correlation_status{};
};

struct ReadbackSubmitSample {
    ReadbackSeq readback_seq{};
    SubmitSeq submit_seq{};
    u64 ready_tick{};
    u64 enqueue_ns{};
    u64 copy_bytes{};
    CmdBufferSeq cmd_buffer_seq{};
    u64 signal_tick{};
};

struct ReadbackReadySample {
    ReadbackSeq readback_seq{};
    SubmitSeq submit_seq{};
    u64 ready_tick{};
    u64 schedule_to_ready_ns{};
    u64 submit_to_ready_ns{};
};

struct ReadbackCommitSample {
    ReadbackSeq readback_seq{};
    FenceSeq fence_seq{};
    ResourceSeq resource_id{};
    ResourceVersion version{};
    VAddr guest_addr{};
    u64 size{};
    u64 memcpy_ns{};
    u64 invalidate_ns{};
    ResourceVersion host_version_before{};
    ResourceVersion host_version_after{};
};

struct ReadbackSourceWatch {
    u64 watch_seq{};
    FenceSeq fence_seq{};
    WaitSeq wait_seq{};
    ReadbackSeq readback_seq{};
    ProducerSeq producer_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    VAddr guest_addr{};
    u64 size{};
    PacketSeq producer_packet{};
    PacketSeq fence_packet{};
    SourceWatchState state{SourceWatchState::Active};
    u64 create_timestamp_ns{};
};

struct ReadbackSourceTerminalSample {
    u64 watch_seq{};
    FenceSeq fence_seq{};
    ReadbackSeq readback_seq{};
    ResourceId source_resource_id{};
    ResourceVersion source_resource_version{};
    TerminalKind terminal_kind{};
    ProducerSeq consumer_producer_seq{};
    PacketSeq consumer_packet_seq{};
    ResourceId consumer_resource_id{};
    ResourceVersion consumer_resource_version{};
    ConsumerAccessPath access_path{ConsumerAccessPath::Unknown};
    VAddr overlap_addr{};
    u64 overlap_size{};
    u32 packets_since_fence{};
    u64 ns_since_fence{};
    CorrelationStatus correlation_status{CorrelationStatus::Complete};
    CmdBufferSeq cmd_buffer_seq{};
    SubmitSeq submit_seq{};
};

struct HostVersionState {
    ResourceVersion version{};
    HostVersionOrigin origin{HostVersionOrigin::Unknown};
    ReadbackSeq origin_readback_seq{};
    ResourceId origin_resource_id{};
    u64 origin_event_seq{};
};

struct GuestSourceConsumeSample {
    ReadbackSeq origin_readback_seq{};
    ResourceId origin_resource_id{};
    ResourceVersion origin_resource_version{};
    VAddr guest_addr{};
    u64 size{};
    ProducerSeq consumer_producer_seq{};
    PacketSeq consumer_packet_seq{};
    ResourceType destination_kind{};
    ResourceId destination_resource_id{};
    GuestSourceConsumePath path{};
};

struct ResourceLineageSample {
    ResourceId source_resource_id{};
    ResourceVersion source_resource_version{};
    ResourceId destination_resource_id{};
    ResourceVersion destination_resource_version{};
    ResourceLineageKind kind{ResourceLineageKind::GpuToGpu};
};

struct ReadWatchInterest {
    FenceSeq fence_seq{};
    FenceGen generation{};
    ReadbackSeq readback_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    VAddr guest_addr{};
    u64 size{};
    ReadWatchKind kind{};
    PacketSeq fence_packet{};
    u64 arm_timestamp_ns{};
};

struct CpuReadObservationSample {
    u32 thread_id{};
    ReadWatchKind watch_kind{};
    FenceSeq fence_seq{};
    FenceGen generation{};
    ReadbackSeq readback_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    VAddr fault_addr{};
    VAddr guest_range_addr{};
    u64 guest_range_size{};
    u64 ns_since_fence{};
    u32 packets_since_fence{};
};

struct SemanticReadFaultSample {
    u32 thread_id{};
    VAddr rip{};
    VAddr fault_addr{};
    u64 access_size{};
    SemanticReadOrigin origin{SemanticReadOrigin::None};
    ReadWatchKind watch_kind{ReadWatchKind::Label};
    FenceSeq fence_seq{};
    ReadbackSeq readback_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    ResourceVersion host_version{};
    HostVersionOrigin host_version_origin{HostVersionOrigin::Unknown};
};

struct SemanticReadUnknownSample {
    u32 thread_id{};
    VAddr rip{};
    VAddr fault_addr{};
    u64 access_size{};
    ReadWatchKind watch_kind{ReadWatchKind::Label};
    FenceSeq fence_seq{};
    ReadbackSeq readback_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
};

enum class SemanticWatchCancelReason : u8 {
    WaitCompleted,
    GenerationSuperseded,
    Observed,
    Overwritten,
    PageConflictWrite,
    Unmap,
    ExplicitCancel,
    CaptureShutdown,
    LivelockBreak,
    Count,
};

struct SemanticWatchCancelSample {
    FenceSeq fence_seq{};
    ReadbackSeq readback_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    VAddr page{};
    VAddr guest_addr{};
    u64 size{};
    ReadWatchKind watch_kind{};
    SemanticWatchCancelReason reason{};
    u32 remaining_page_owners{};
};

struct SemanticPageConflictWriteSample {
    u32 thread_id{};
    VAddr rip{};
    VAddr fault_addr{};
    u64 write_size{};
    VAddr watched_addr{};
    u64 watched_size{};
    ReadWatchKind watch_kind{};
    FenceSeq fence_seq{};
    ReadbackSeq readback_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
};

struct ResourceBarrierLinkSample {
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    FenceSeq fence_seq{};
    ReadbackSeq readback_seq{};
    CmdBufferSeq cmd_buffer_seq{};
    SubmitSeq submit_seq{};
    u32 old_layout{};
    u32 new_layout{};
    u64 src_stage{};
    u64 src_access{};
    u64 dst_stage{};
    u64 dst_access{};
    u64 subresource_or_range{};
    const char* reason_path{"Unknown"};
};

struct AcquireMemSample {
    PacketSeq packet_seq{};
    FrameSeq frame_seq{};
    u32 queue_id{};
    Pm4Engine engine{};
    u32 cp_coher_cntl{};
    u32 cp_coher_size_lo{};
    u32 cp_coher_size_hi{};
    u32 cp_coher_base_lo{};
    u32 cp_coher_base_hi{};
    u32 poll_interval{};
    VAddr decoded_base_addr{};
    u64 decoded_size{};
    u32 decoded_cache_flags{};
    WaitSeq previous_wait_seq{};
    FenceSeq shadow_fence_seq{};
};

struct FenceSignalSample {
    FenceSeq fence_seq{};
    FenceGen generation{};
    VAddr label_addr{};
    u64 label_value{};
    u64 ready_tick{};
    u64 signal_ns{};
    u32 writeback_count{};
    u64 writeback_bytes{};
    u64 gpu_ready_to_signal_ns{};
    u32 irq{};
    MemoryWriteOrigin signal_origin{MemoryWriteOrigin::FenceSignal};
};

struct HostWaitSample {
    WaitSeq wait_seq{};
    HostWaitReason reason{};
    u64 requested_tick{};
    u64 current_gpu_tick_before{};
    u64 cpu_tick{};
    u64 queue_depth_estimate{};
    u64 start_ns{};
    u64 duration_ns{};
    FenceSeq fence_seq{};
    ReadbackSeq readback_seq{};
    SubmitSeq submit_seq{};
    PacketSeq packet_seq{};
};

struct SubmitRecordSample {
    SubmitSeq submit_seq{};
    FrameSeq frame_seq{};
    SubmitReason reason{};
    u64 signal_tick{};
    u32 command_count{};
    u32 producer_count{};
    u32 copy_count{};
    u64 copy_bytes{};
    u32 pending_fence_count{};
    u32 pending_readback_count{};
    u64 cpu_ahead_ticks{};
    u64 gpu_completed_tick{};
    u32 scheduler_id{};
    u32 queue_role{};
    CmdBufferSeq cmd_buffer_seq{};
};

struct ProducerRecordSample {
    ProducerSeq producer_seq{};
    PacketSeq packet_seq{};
    FrameSeq frame_seq{};
    ProducerClass producer_type{};
    u32 queue_id{};
    u32 stage{};
    u64 shader_hash{};
    u64 pipeline_hash{};
    u32 dispatch_x{};
    u32 dispatch_y{};
    u32 dispatch_z{};
    u32 draw_count{};
    u64 start_ns{};
    u64 duration_ns{};
    u32 write_range_count{};
    u64 write_bytes{};
    u32 write_resource_count{};
    bool is_promoted{};
    CmdBufferSeq cmd_buffer_seq{};
    SubmitSeq submit_seq{};
};

struct ShadowFencePolicySample {
    FenceSeq fence_seq{};
    FenceClassification would_classify{};
    bool would_skip_host_readback{};
    u32 reason_bits{};
    WaitConfidence confidence{};
};

struct TraceGapSample {
    u32 stream_id{};
    u64 first_missing_seq{};
    u64 last_missing_seq{};
    TraceGapReason reason{};
    u64 count{};
};

struct RingHealthSample {
    u32 stream_id{};
    u64 capacity_records{};
    u64 capacity_bytes{};
    u64 seen{};
    u64 emitted{};
    u64 filtered{};
    u64 sampled{};
    u64 written{};
    u64 overwritten{};
    u64 dropped_writer_backpressure{};
    u64 first_event_seq{};
    u64 last_event_seq{};
    u64 first_timestamp{};
    u64 last_timestamp{};
    u64 oldest_retained_timestamp{};
};

struct FastpathCandidateSample {
    u64 candidate_seq{};
    u64 timestamp_ns{};
    u32 title_id_hash{};
    ProducerSeq producer_seq{};
    PacketSeq producer_packet_seq{};
    u32 producer_kind{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    u32 image_id{};
    u64 image_uid{};
    VAddr guest_addr{};
    u32 size{};
    FenceSeq fence_seq{};
    PacketSeq eos_packet_seq{};
    VAddr label_addr{};
    u32 label_value{};
    u32 label_num_bytes{};
    PacketSeq wait_packet_seq{};
    u32 wait_compare{};
    u32 wait_ref{};
    u32 wait_mask{};
    PacketSeq acquire_packet_seq{};
    u32 acquire_raw_cntl{};
    FastpathEligibility eligibility{FastpathEligibility::Eligible};
    FastpathRejectReason reject_reason{FastpathRejectReason::None};
};

struct GpuAuthorityCreateSample {
    u64 authority_seq{};
    u64 candidate_seq{};
    u64 timestamp_ns{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    u32 image_id{};
    u64 image_uid{};
    VAddr guest_begin{};
    VAddr guest_end{};
    u32 size{};
    ProducerSeq producer_seq{};
    PacketSeq producer_packet_seq{};
    u64 producer_tick{};
    CmdBufferSeq cmd_buffer_seq{};
    SubmitSeq submit_seq{};
    FenceSeq fence_seq{};
    u64 virtual_fence_seq{};
    VAddr label_addr{};
    u64 label_generation{};
};

struct VirtualFenceCreateSample {
    u64 virtual_fence_seq{};
    u64 authority_seq{};
    FenceSeq fence_seq{};
    VAddr label_addr{};
    u64 label_generation{};
    u32 expected_value{};
    u64 producer_tick{};
    PacketSeq producer_packet_seq{};
    PacketSeq eos_packet_seq{};
    PacketSeq wait_packet_seq{};
    PacketSeq acquire_packet_seq{};
};

struct VirtualWaitConsumeSample {
    u64 virtual_fence_seq{};
    u64 authority_seq{};
    WaitSeq wait_seq{};
    PacketSeq wait_packet_seq{};
    VAddr label_addr{};
    u64 label_generation{};
    u64 producer_tick{};
    u8 producer_tick_complete_at_consume{};
    VirtualWaitResult result{VirtualWaitResult::Virtualized};
};

struct AsyncLabelSignalSample {
    u64 virtual_fence_seq{};
    u64 authority_seq{};
    VAddr label_addr{};
    u64 scheduled_generation{};
    u64 current_generation{};
    u64 producer_tick{};
    u64 current_completed_tick{};
    AsyncLabelAction action{AsyncLabelAction::Wrote};
};

struct AuthorityGpuConsumeSample {
    u64 authority_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    u32 image_id{};
    u64 image_uid{};
    u64 consumer_seq{};
    PacketSeq consumer_packet_seq{};
    u32 consumer_kind{};
    u32 requested_access{};
    u32 requested_layout{};
    u64 producer_tick{};
    CmdBufferSeq consumer_cmd_buffer_seq{};
    SubmitSeq consumer_submit_seq{};
};

struct AuthorityBarrierValidationSample {
    u64 authority_seq{};
    u64 consumer_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    u32 old_layout{};
    u32 new_layout{};
    u64 src_stage{};
    u64 src_access{};
    u64 dst_stage{};
    u64 dst_access{};
    u64 subresource_range{};
    u8 valid_write_dependency{};
};

struct AuthorityRamDemandSample {
    u64 ram_demand_seq{};
    u64 ram_demand_group_seq{};
    u64 timestamp_ns{};
    u64 authority_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    VAddr authority_begin{};
    VAddr authority_end{};
    VAddr request_addr{};
    u64 request_size{};
    VAddr overlap_begin{};
    u64 overlap_size{};
    GuestSourceConsumePath path{GuestSourceConsumePath::StreamBufferCopy};
    u64 consumer_seq{};
    ProducerSeq consumer_producer_seq{};
    PacketSeq consumer_packet_seq{};
    ResourceType destination_kind{ResourceType::Buffer};
    ResourceId destination_resource_id{};
    u8 authority_state_before{};
};

struct LazyMaterializeBeginSample {
    u64 materialize_seq{};
    u64 ram_demand_seq{};
    u64 authority_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    u32 image_id{};
    u64 image_uid{};
    u64 producer_tick{};
    u64 current_completed_tick{};
    VAddr guest_begin{};
    VAddr guest_end{};
    u32 reason{};
};

struct LazyMaterializeEndSample {
    u64 materialize_seq{};
    u64 ram_demand_seq{};
    u64 authority_seq{};
    u64 producer_tick{};
    u64 completed_tick{};
    u64 bytes_materialized{};
    VAddr guest_begin{};
    VAddr guest_end{};
    LazyMaterializeResult result{LazyMaterializeResult::Success};
    u8 host_current_after{1};
    s8 validation_bytes_equal{1};
};

struct AuthorityRamConsumeSample {
    u64 ram_demand_seq{};
    u64 authority_seq{};
    u64 materialize_seq{};
    u64 consumer_seq{};
    GuestSourceConsumePath path{GuestSourceConsumePath::StreamBufferCopy};
    VAddr request_addr{};
    u64 request_size{};
    VAddr overlap_begin{};
    u64 overlap_size{};
    u8 host_current{1};
    u8 producer_tick_complete{1};
    u8 materialize_success{1};
};

struct AuthorityCpuReadSample {
    u64 authority_seq{};
    VAddr fault_addr{};
    VAddr guest_read_begin{};
    u64 guest_read_size{};
    SemanticReadOrigin origin{SemanticReadOrigin::GuestDirect};
    u64 materialize_seq{};
    u8 resumed_after_materialize{1};
};

struct AuthoritySupersedeSample {
    u64 old_authority_seq{};
    u64 new_authority_seq{};
    VAddr overlap_begin{};
    u64 overlap_size{};
    ResourceId old_resource_id{};
    ResourceVersion old_resource_version{};
    ResourceId new_resource_id{};
    ResourceVersion new_resource_version{};
    u8 old_host_current{};
};

struct FastpathFallbackSample {
    u64 candidate_seq{};
    u64 authority_seq_if_created{};
    FastpathFallbackPhase phase{FastpathFallbackPhase::Recognizer};
    FastpathFallbackReason reason{FastpathFallbackReason::None};
};

struct ConservativeDownloadDecisionSample {
    u64 timestamp_ns{};
    u32 trigger{};
    FenceSeq fence_seq{};
    u32 image_id{};
    u64 image_uid{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    VAddr guest_addr{};
    u32 size{};
    u64 authority_seq{};
    u8 authority_state{};
    ConservativeDownloadDecision decision{ConservativeDownloadDecision::LegacyDownload};
    u32 reason{};
    u8 readback_schedule_seen{};
    u64 readback_seq{};
};

struct AuthorityConservativeReadbackSuppressedSample {
    u64 authority_seq{};
    ResourceId resource_id{};
    ResourceVersion resource_version{};
    u32 trigger{};
    FenceSeq fence_seq{};
    VAddr guest_addr{};
    u32 size{};
    u32 image_id{};
    u64 image_uid{};
};

struct AuthorityHostMaterializeRequiredSample {
    u64 authority_seq{};
    u32 reason{};
    VAddr request_addr{};
    u64 request_size{};
    VAddr overlap_addr{};
    u64 overlap_size{};
};

struct FastpathWaitDecisionSample {
    u64 candidate_seq{};
    u64 virtual_fence_seq{};
    u64 authority_seq{};
    WaitSeq wait_seq{};
    PacketSeq wait_packet_seq{};
    VAddr label_addr{};
    u64 label_generation{};
    u32 ref{};
    u32 mask{};
    u32 compare{};
    u64 producer_tick{};
    u64 current_tick{};
    FastpathWaitDecision decision{FastpathWaitDecision::Virtualized};
    u32 reason{};
};

struct VirtualFenceForcedCompletionSample {
    u64 virtual_fence_seq{};
    u64 authority_seq{};
    VirtualFenceForcedCompletionReason reason{VirtualFenceForcedCompletionReason::HostSideEffect};
    u64 producer_tick{};
    u8 was_submitted{};
    u8 waited{};
    u64 duration_ns{};
};

struct CpuToGpuLabelWaitSample {
    WaitSeq wait_seq{};
    FrameSeq frame_id{};
    VAddr label_addr{};
    u32 ref{};
    u32 mask{};
    u64 wait_begin{};
    u64 wait_end{};
    u64 duration_ns{};
    u64 last_guest_write_ts{};
    u32 last_guest_write_value{};
    u64 writer_thread_id{};
    u64 delta_write_to_wait_complete_ns{};
    u32 yield_count{};
    u32 wait_progress_submit_count{};
    u64 gpu_idle_overlap_ns{};
};

class TelemetryProducerScope {
public:
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    TelemetryProducerScope(ProducerSeq seq, PacketSeq pkt, ProducerClass type, u32 queue) noexcept;
    ~TelemetryProducerScope();
#else
    TelemetryProducerScope(ProducerSeq seq, PacketSeq pkt, ProducerClass type, u32 queue) noexcept {
        static_cast<void>(seq);
        static_cast<void>(pkt);
        static_cast<void>(type);
        static_cast<void>(queue);
    }
    ~TelemetryProducerScope() = default;
#endif

private:
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    bool active{false};
#endif
};

class TelemetryMemoryWriteScope {
public:
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    explicit TelemetryMemoryWriteScope(GuestMemoryWriteOrigin origin, FenceSeq fence_seq = 0) noexcept;
    ~TelemetryMemoryWriteScope();
#else
    explicit TelemetryMemoryWriteScope(GuestMemoryWriteOrigin origin, FenceSeq fence_seq = 0) noexcept {
        static_cast<void>(origin);
        static_cast<void>(fence_seq);
    }
    ~TelemetryMemoryWriteScope() = default;
#endif

private:
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    GuestMemoryWriteOrigin prev_origin{GuestMemoryWriteOrigin::GuestCpu};
    FenceSeq prev_fence{0};
#endif
};

using ScopedMemoryWriteOrigin = TelemetryMemoryWriteScope;

inline thread_local FenceTraceToken tl_active_fence_cause{};

class ScopedFenceCause {
public:
    explicit ScopedFenceCause(const FenceTraceToken& token) noexcept
        : prev_token{tl_active_fence_cause} {
        tl_active_fence_cause = token;
    }
    ~ScopedFenceCause() {
        tl_active_fence_cause = prev_token;
    }

private:
    FenceTraceToken prev_token;
};

inline FenceTraceToken CurrentFenceCause() noexcept {
    return tl_active_fence_cause;
}

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
EventSeq NextEventSeqEnabled() noexcept;
PacketSeq NextPacketSeqEnabled() noexcept;
ProducerSeq NextProducerSeqEnabled() noexcept;
FenceSeq NextFenceSeqEnabled() noexcept;
FenceGen NextFenceGenEnabled() noexcept;
WaitSeq NextWaitSeqEnabled() noexcept;
ResourceSeq NextResourceSeqEnabled() noexcept;
ResourceVersion NextResourceVersionEnabled() noexcept;
SubmitSeq NextSubmitSeqEnabled() noexcept;
CpuAccessSeq NextCpuAccessSeqEnabled() noexcept;
ReadbackSeq NextReadbackSeqEnabled() noexcept;
CmdBufferSeq NextCmdBufferSeqEnabled() noexcept;

FrameSeq CurrentFrameSeqEnabled() noexcept;
CmdBufferSeq CurrentCmdBufferSeqEnabled() noexcept;
PacketSeq CurrentPacketSeqEnabled() noexcept;
ProducerSeq CurrentProducerSeqEnabled() noexcept;

void RecordSyncPm4PacketEnabled(const SyncPm4PacketSample& sample) noexcept;
void RecordProducerBeginEnabled(const ProducerBeginSample& sample) noexcept;
void RecordProducerEndEnabled(const ProducerEndSample& sample) noexcept;
void RecordProducerRecordEnabled(const ProducerRecordSample& sample) noexcept;
void RecordResourceWriteEnabled(const ResourceWriteSample& sample) noexcept;
void RecordFenceCreateEnabled(const FenceCreateSample& sample) noexcept;
void RecordFenceEpochLinkEnabled(const FenceEpochLinkSample& sample) noexcept;
void RecordFenceMatchAttemptEnabled(const FenceMatchAttemptSample& sample) noexcept;
void RecordFenceMatchDiagnosticEnabled(const FenceMatchDiagnosticSample& sample) noexcept;
void RecordWaitCreateEnabled(const WaitCreateSample& sample) noexcept;
void RecordWaitCompleteEnabled(const WaitCompleteSample& sample) noexcept;
void RecordFirstConsumerEnabled(const FirstConsumerSample& sample) noexcept;
void RecordFenceClassificationEnabled(const FenceClassificationSample& sample) noexcept;
void RecordCpuMemoryAccessEnabled(const CpuAccessSample& sample) noexcept;
void RecordCpuLabelAccessEnabled(const CpuLabelAccessSample& sample) noexcept;
void RecordCpuReadRequiresMaterializationEnabled(const CpuMaterializationSample& sample) noexcept;
void RecordStaleGuestSourceAttemptEnabled(const StaleGuestAttemptSample& sample) noexcept;
void RecordGpuAliasMaterializeEnabled(const GpuAliasMaterializeSample& sample) noexcept;
void RecordResourceEpochPromotedEnabled(const ResourceEpochPromotedSample& sample) noexcept;
void RecordFenceResourceLinkEnabled(const FenceResourceLinkSample& sample) noexcept;
void RecordReadbackScheduleEnabled(const ReadbackScheduleSample& sample) noexcept;
void RecordReadbackSubmitEnabled(const ReadbackSubmitSample& sample) noexcept;
void RecordReadbackReadyEnabled(const ReadbackReadySample& sample) noexcept;
void RecordReadbackCommitEnabled(const ReadbackCommitSample& sample) noexcept;
void RecordFenceSignalEnabled(const FenceSignalSample& sample) noexcept;
void RecordHostWaitEnabled(const HostWaitSample& sample) noexcept;
void RecordSubmitRecordEnabled(const SubmitRecordSample& sample) noexcept;
void RecordShadowFencePolicyEnabled(const ShadowFencePolicySample& sample) noexcept;
void RecordTraceGapEnabled(const TraceGapSample& sample) noexcept;
void RecordRingHealthEnabled(const RingHealthSample& sample) noexcept;

FenceSeq MatchFenceForWaitEnabled(WaitSeq wait_seq, PacketSeq packet_seq, FrameSeq frame_seq,
                                  u32 queue_id, Pm4Engine engine, VAddr wait_addr, u32 ref,
                                  u32 mask, u32 function, PacketSeq prev_pkt, u32 prev_op) noexcept;
void ArmConsumerProbeEnabled(FenceSeq fence_seq, WaitSeq wait_seq, PacketSeq wait_pkt,
                             ConsumerProbeKind probe_kind = ConsumerProbeKind::MatchedFence) noexcept;
void CheckConsumerOverlapEnabled(ProducerSeq consumer_prod, PacketSeq pkt, ProducerClass consumer_type,
                                 VAddr addr, u64 size, ConsumerAccessPath access_path = ConsumerAccessPath::Unknown,
                                 ResourceSeq consumer_res_id = 0, ResourceVersion consumer_res_ver = 0,
                                 ConsumerConfidence confidence = ConsumerConfidence::ExactResourceAndVersion,
                                 u64 pipeline_hash = 0, u64 shader_hash = 0) noexcept;
void AdvanceConsumerProbesEnabled(PacketSeq current_packet, bool is_present) noexcept;
GuestMemoryWriteOrigin CurrentMemoryWriteOriginEnabled() noexcept;
void SetCurrentMemoryWriteOriginEnabled(GuestMemoryWriteOrigin origin, FenceSeq fence_seq) noexcept;
void SetCurrentProducerScopeEnabled(ProducerSeq seq, PacketSeq pkt, ProducerClass type, u32 queue) noexcept;
void ClearCurrentProducerScopeEnabled() noexcept;
void RegisterSubmitTickEnabled(u64 signal_tick, SubmitSeq submit_seq) noexcept;
SubmitSeq LookupSubmitSeqEnabled(u64 signal_tick) noexcept;
WatchSeq NextWatchSeqEnabled() noexcept;

void RecordReadbackSourceTerminalEnabled(const ReadbackSourceTerminalSample& sample) noexcept;
void RecordGuestSourceConsumeEnabled(const GuestSourceConsumeSample& sample) noexcept;
void RecordResourceLineageEnabled(const ResourceLineageSample& sample) noexcept;
void RecordCpuReadObservationEnabled(const CpuReadObservationSample& sample) noexcept;
void RecordSemanticReadFaultEnabled(const SemanticReadFaultSample& sample) noexcept;
void RecordSemanticReadUnknownEnabled(const SemanticReadUnknownSample& sample) noexcept;
void RecordResourceBarrierLinkEnabled(const ResourceBarrierLinkSample& sample) noexcept;
void RecordAcquireMemEnabled(const AcquireMemSample& sample) noexcept;

void ArmReadbackSourceWatchEnabled(const ReadbackSourceWatch& watch) noexcept;
void ResolveReadbackSourceWatchEnabled(ResourceId res_id, ResourceVersion ver, VAddr addr, u64 size,
                                       TerminalKind kind, ProducerSeq consumer_prod = 0,
                                       PacketSeq consumer_pkt = 0, ResourceId consumer_res = 0,
                                       ResourceVersion consumer_ver = 0,
                                       ConsumerAccessPath path = ConsumerAccessPath::Unknown,
                                       CmdBufferSeq cmd_buf = 0, SubmitSeq submit_seq = 0) noexcept;
bool HasActiveReadbackSourceWatchEnabled(ResourceId res_id, ResourceVersion ver) noexcept;
ReadbackSourceWatch GetActiveReadbackSourceWatchEnabled(ResourceId res_id, ResourceVersion ver) noexcept;
void UpdateHostVersionEnabled(VAddr addr, u64 size, ResourceVersion ver, HostVersionOrigin origin,
                              ReadbackSeq readback_seq, ResourceId res_id) noexcept;
void CheckGuestSourceConsumeEnabled(VAddr addr, u64 size, ProducerSeq prod_seq, PacketSeq pkt_seq,
                                    ResourceType dst_kind, ResourceId dst_id,
                                    GuestSourceConsumePath path) noexcept;
TraceCaptureProfile GetCaptureProfileEnabled() noexcept;
void ArmReadWatchInterestEnabled(const ReadWatchInterest& interest) noexcept;
void RecordSemanticWatchCancelEnabled(const SemanticWatchCancelSample& sample) noexcept;
void RecordSemanticPageConflictWriteEnabled(const SemanticPageConflictWriteSample& sample) noexcept;
void SubtractEnabled(Counter counter, u64 value = 1) noexcept;
using DisarmWatchCallback = void (*)(void* user_data, VAddr addr, u64 size) noexcept;
void DisarmLabelReadWatchEnabled(FenceSeq fence_seq, VAddr guest_addr, u64 size,
                                 SemanticWatchCancelReason reason = SemanticWatchCancelReason::ExplicitCancel,
                                 DisarmWatchCallback disarm_cb = nullptr,
                                 void* user_data = nullptr) noexcept;
void HandleWriteFaultOnWatchedPageEnabled(VAddr addr, u64 size, u32 thread_id, VAddr rip = 0,
                                         DisarmWatchCallback disarm_cb = nullptr,
                                         void* user_data = nullptr) noexcept;
bool CheckCpuReadObservationEnabled(VAddr addr, u64 size, u32 thread_id, VAddr rip = 0,
                                    DisarmWatchCallback disarm_cb = nullptr,
                                    void* user_data = nullptr) noexcept;
void RegisterCmdBufferSubmitEnabled(CmdBufferSeq cmd_buf, SubmitSeq submit_seq) noexcept;
SubmitSeq LookupCmdBufferSubmitEnabled(CmdBufferSeq cmd_buf) noexcept;
void RegisterPendingReadbackForSubmitEnabled(ReadbackSeq readback_seq, CmdBufferSeq cmd_buf) noexcept;
void PromotePendingReadbacksOnSubmitEnabled(CmdBufferSeq cmd_buf, SubmitSeq submit_seq,
                                           u64 signal_tick) noexcept;

void RecordPm4ControlEnabled(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                             u32 control0, u32 control1, u32 tag = 0) noexcept;
void RecordPm4RegisterEnabled(Pm4Engine engine, u32 opcode, u32 register_offset, u32 words,
                              bool changed) noexcept;
void RecordPm4WaitEnabled(Pm4Engine engine, u32 queue_id, u32 depth, u32 control,
                          u64 location, u32 reference, u32 mask, u32 poll_interval,
                          u64 failed_tests, bool vo_sleep) noexcept;

void RecordFastpathCandidateEnabled(const FastpathCandidateSample& sample) noexcept;
void RecordGpuAuthorityCreateEnabled(const GpuAuthorityCreateSample& sample) noexcept;
void RecordVirtualFenceCreateEnabled(const VirtualFenceCreateSample& sample) noexcept;
void RecordVirtualWaitConsumeEnabled(const VirtualWaitConsumeSample& sample) noexcept;
void RecordAsyncLabelSignalEnabled(const AsyncLabelSignalSample& sample) noexcept;
void RecordAuthorityGpuConsumeEnabled(const AuthorityGpuConsumeSample& sample) noexcept;
void RecordAuthorityBarrierValidationEnabled(const AuthorityBarrierValidationSample& sample) noexcept;
void RecordAuthorityRamDemandEnabled(const AuthorityRamDemandSample& sample) noexcept;
void RecordLazyMaterializeBeginEnabled(const LazyMaterializeBeginSample& sample) noexcept;
void RecordLazyMaterializeEndEnabled(const LazyMaterializeEndSample& sample) noexcept;
void RecordAuthorityRamConsumeEnabled(const AuthorityRamConsumeSample& sample) noexcept;
void RecordAuthorityCpuReadEnabled(const AuthorityCpuReadSample& sample) noexcept;
void RecordAuthoritySupersedeEnabled(const AuthoritySupersedeSample& sample) noexcept;
void RecordFastpathFallbackEnabled(const FastpathFallbackSample& sample) noexcept;
void RecordConservativeDownloadDecisionEnabled(const ConservativeDownloadDecisionSample& sample) noexcept;
void RecordAuthorityConservativeReadbackSuppressedEnabled(const AuthorityConservativeReadbackSuppressedSample& sample) noexcept;
void RecordAuthorityHostMaterializeRequiredEnabled(const AuthorityHostMaterializeRequiredSample& sample) noexcept;
void RecordFastpathWaitDecisionEnabled(const FastpathWaitDecisionSample& sample) noexcept;
void RecordVirtualFenceForcedCompletionEnabled(const VirtualFenceForcedCompletionSample& sample) noexcept;
void RecordCpuToGpuLabelWaitEnabled(const CpuToGpuLabelWaitSample& sample) noexcept;
void RecordGuestCpuLabelWriteEnabled(VAddr addr, u32 val, u64 timestamp, u64 thread_id) noexcept;

u64 NextCandidateSeqEnabled() noexcept;
u64 NextAuthoritySeqEnabled() noexcept;
u64 NextVirtualFenceSeqEnabled() noexcept;
u64 NextRamDemandSeqEnabled() noexcept;
u64 NextRamDemandGroupSeqEnabled() noexcept;
u64 NextMaterializeSeqEnabled() noexcept;
u64 NextConsumerSeqEnabled() noexcept;
#endif

inline EventSeq NextEventSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextEventSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline PacketSeq NextPacketSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextPacketSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline ProducerSeq NextProducerSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextProducerSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline FenceSeq NextFenceSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextFenceSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline FenceGen NextFenceGen() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextFenceGenEnabled() : 0;
#else
    return 0;
#endif
}

inline WaitSeq NextWaitSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextWaitSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline ResourceSeq NextResourceSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextResourceSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline ResourceVersion NextResourceVersion() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextResourceVersionEnabled() : 0;
#else
    return 0;
#endif
}

inline SubmitSeq NextSubmitSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextSubmitSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline CpuAccessSeq NextCpuAccessSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextCpuAccessSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline ReadbackSeq NextReadbackSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextReadbackSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline u64 NextCandidateSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextCandidateSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline u64 NextAuthoritySeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextAuthoritySeqEnabled() : 0;
#else
    return 0;
#endif
}

inline u64 NextVirtualFenceSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextVirtualFenceSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline u64 NextRamDemandSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextRamDemandSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline u64 NextRamDemandGroupSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextRamDemandGroupSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline u64 NextMaterializeSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextMaterializeSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline u64 NextConsumerSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextConsumerSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline CmdBufferSeq NextCmdBufferSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextCmdBufferSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline FrameSeq CurrentFrameSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? CurrentFrameSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline CmdBufferSeq CurrentCmdBufferSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? CurrentCmdBufferSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline PacketSeq CurrentPacketSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? CurrentPacketSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline ProducerSeq CurrentProducerSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? CurrentProducerSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline void RecordSyncPm4Packet(const SyncPm4PacketSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordSyncPm4PacketEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordProducerBegin(const ProducerBeginSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordProducerBeginEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordProducerEnd(const ProducerEndSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordProducerEndEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordProducerRecord(const ProducerRecordSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordProducerRecordEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordResourceWrite(const ResourceWriteSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordResourceWriteEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFenceCreate(const FenceCreateSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFenceCreateEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFenceEpochLink(const FenceEpochLinkSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFenceEpochLinkEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFenceMatchAttempt(const FenceMatchAttemptSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFenceMatchAttemptEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFenceMatchDiagnostic(const FenceMatchDiagnosticSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFenceMatchDiagnosticEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordWaitCreate(const WaitCreateSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordWaitCreateEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordWaitComplete(const WaitCompleteSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordWaitCompleteEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFirstConsumer(const FirstConsumerSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFirstConsumerEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFenceClassification(const FenceClassificationSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFenceClassificationEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordCpuMemoryAccess(const CpuAccessSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordCpuMemoryAccessEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordCpuLabelAccess(const CpuLabelAccessSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordCpuLabelAccessEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordCpuReadRequiresMaterialization(const CpuMaterializationSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordCpuReadRequiresMaterializationEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordStaleGuestSourceAttempt(const StaleGuestAttemptSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordStaleGuestSourceAttemptEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordGpuAliasMaterialize(const GpuAliasMaterializeSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordGpuAliasMaterializeEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordResourceEpochPromoted(const ResourceEpochPromotedSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordResourceEpochPromotedEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFenceResourceLink(const FenceResourceLinkSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFenceResourceLinkEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordReadbackSchedule(const ReadbackScheduleSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordReadbackScheduleEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordReadbackSubmit(const ReadbackSubmitSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordReadbackSubmitEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordReadbackReady(const ReadbackReadySample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordReadbackReadyEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordReadbackCommit(const ReadbackCommitSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordReadbackCommitEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFenceSignal(const FenceSignalSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFenceSignalEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordHostWait(const HostWaitSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordHostWaitEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordSubmitRecord(const SubmitRecordSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordSubmitRecordEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordShadowFencePolicy(const ShadowFencePolicySample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordShadowFencePolicyEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordTraceGap(const TraceGapSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordTraceGapEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordRingHealth(const RingHealthSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordRingHealthEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline WatchSeq NextWatchSeq() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? NextWatchSeqEnabled() : 0;
#else
    return 0;
#endif
}

inline void RecordReadbackSourceTerminal(const ReadbackSourceTerminalSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordReadbackSourceTerminalEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordGuestSourceConsume(const GuestSourceConsumeSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordGuestSourceConsumeEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordResourceLineage(const ResourceLineageSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordResourceLineageEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordCpuReadObservation(const CpuReadObservationSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordCpuReadObservationEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordSemanticReadFault(const SemanticReadFaultSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordSemanticReadFaultEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordSemanticReadUnknown(const SemanticReadUnknownSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordSemanticReadUnknownEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordResourceBarrierLink(const ResourceBarrierLinkSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordResourceBarrierLinkEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAcquireMem(const AcquireMemSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAcquireMemEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFastpathCandidate(const FastpathCandidateSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFastpathCandidateEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordGpuAuthorityCreate(const GpuAuthorityCreateSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordGpuAuthorityCreateEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordVirtualFenceCreate(const VirtualFenceCreateSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordVirtualFenceCreateEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordVirtualWaitConsume(const VirtualWaitConsumeSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordVirtualWaitConsumeEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAsyncLabelSignal(const AsyncLabelSignalSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAsyncLabelSignalEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthorityGpuConsume(const AuthorityGpuConsumeSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthorityGpuConsumeEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthorityBarrierValidation(const AuthorityBarrierValidationSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthorityBarrierValidationEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthorityRamDemand(const AuthorityRamDemandSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthorityRamDemandEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordLazyMaterializeBegin(const LazyMaterializeBeginSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordLazyMaterializeBeginEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordLazyMaterializeEnd(const LazyMaterializeEndSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordLazyMaterializeEndEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthorityRamConsume(const AuthorityRamConsumeSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthorityRamConsumeEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthorityCpuRead(const AuthorityCpuReadSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthorityCpuReadEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthoritySupersede(const AuthoritySupersedeSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthoritySupersedeEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFastpathFallback(const FastpathFallbackSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFastpathFallbackEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordConservativeDownloadDecision(const ConservativeDownloadDecisionSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordConservativeDownloadDecisionEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthorityConservativeReadbackSuppressed(const AuthorityConservativeReadbackSuppressedSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthorityConservativeReadbackSuppressedEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordAuthorityHostMaterializeRequired(const AuthorityHostMaterializeRequiredSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordAuthorityHostMaterializeRequiredEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordFastpathWaitDecision(const FastpathWaitDecisionSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFastpathWaitDecisionEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordVirtualFenceForcedCompletion(const VirtualFenceForcedCompletionSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordVirtualFenceForcedCompletionEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordCpuToGpuLabelWait(const CpuToGpuLabelWaitSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordCpuToGpuLabelWaitEnabled(sample);
    }
#else
    static_cast<void>(sample);
#endif
}

inline void RecordGuestCpuLabelWrite(VAddr addr, u32 val, u64 timestamp, u64 thread_id) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordGuestCpuLabelWriteEnabled(addr, val, timestamp, thread_id);
    }
#else
    static_cast<void>(addr);
    static_cast<void>(val);
    static_cast<void>(timestamp);
    static_cast<void>(thread_id);
#endif
}

inline void ArmReadbackSourceWatch(const ReadbackSourceWatch& watch) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        ArmReadbackSourceWatchEnabled(watch);
    }
#else
    static_cast<void>(watch);
#endif
}

inline void ResolveReadbackSourceWatch(ResourceId res_id, ResourceVersion ver, VAddr addr, u64 size,
                                      TerminalKind kind, ProducerSeq consumer_prod = 0,
                                      PacketSeq consumer_pkt = 0, ResourceId consumer_res = 0,
                                      ResourceVersion consumer_ver = 0,
                                      ConsumerAccessPath path = ConsumerAccessPath::Unknown,
                                      CmdBufferSeq cmd_buf = 0, SubmitSeq submit_seq = 0) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        ResolveReadbackSourceWatchEnabled(res_id, ver, addr, size, kind, consumer_prod,
                                          consumer_pkt, consumer_res, consumer_ver, path, cmd_buf,
                                          submit_seq);
    }
#else
    static_cast<void>(res_id);
    static_cast<void>(ver);
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(kind);
    static_cast<void>(consumer_prod);
    static_cast<void>(consumer_pkt);
    static_cast<void>(consumer_res);
    static_cast<void>(consumer_ver);
    static_cast<void>(path);
    static_cast<void>(cmd_buf);
    static_cast<void>(submit_seq);
#endif
}

inline bool HasActiveReadbackSourceWatch(ResourceId res_id, ResourceVersion ver) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? HasActiveReadbackSourceWatchEnabled(res_id, ver) : false;
#else
    static_cast<void>(res_id);
    static_cast<void>(ver);
    return false;
#endif
}

inline ReadbackSourceWatch GetActiveReadbackSourceWatch(ResourceId res_id, ResourceVersion ver) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? GetActiveReadbackSourceWatchEnabled(res_id, ver) : ReadbackSourceWatch{};
#else
    static_cast<void>(res_id);
    static_cast<void>(ver);
    return {};
#endif
}

inline void UpdateHostVersion(VAddr addr, u64 size, ResourceVersion ver, HostVersionOrigin origin,
                             ReadbackSeq readback_seq = 0, ResourceId res_id = 0) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        UpdateHostVersionEnabled(addr, size, ver, origin, readback_seq, res_id);
    }
#else
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(ver);
    static_cast<void>(origin);
    static_cast<void>(readback_seq);
    static_cast<void>(res_id);
#endif
}

inline void CheckGuestSourceConsume(VAddr addr, u64 size, ProducerSeq prod_seq, PacketSeq pkt_seq,
                                   ResourceType dst_kind, ResourceId dst_id,
                                   GuestSourceConsumePath path) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        CheckGuestSourceConsumeEnabled(addr, size, prod_seq, pkt_seq, dst_kind, dst_id, path);
    }
#else
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(prod_seq);
    static_cast<void>(pkt_seq);
    static_cast<void>(dst_kind);
    static_cast<void>(dst_id);
    static_cast<void>(path);
#endif
}

inline TraceCaptureProfile GetCaptureProfile() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? GetCaptureProfileEnabled() : TraceCaptureProfile::SyncPerf;
#else
    return TraceCaptureProfile::SyncPerf;
#endif
}

inline bool IsSyncSemanticProfile() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() && (GetCaptureProfile() == TraceCaptureProfile::SyncSemantic);
#else
    return false;
#endif
}

inline bool IsFastpathValidationProfile() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() && (GetCaptureProfile() == TraceCaptureProfile::SyncFastpathValidation ||
                         GetCaptureProfile() == TraceCaptureProfile::SyncSemantic);
#else
    return false;
#endif
}

inline void ArmReadWatchInterest(const ReadWatchInterest& interest) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        ArmReadWatchInterestEnabled(interest);
    }
#else
    static_cast<void>(interest);
#endif
}

template <typename Func>
inline void DisarmLabelReadWatch(FenceSeq fence_seq, VAddr guest_addr, u64 size,
                                 SemanticWatchCancelReason reason, Func&& disarm_func) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        auto cb = [](void* ud, VAddr a, u64 s) noexcept {
            (*static_cast<std::remove_reference_t<Func>*>(ud))(a, s);
        };
        DisarmLabelReadWatchEnabled(fence_seq, guest_addr, size, reason, cb, std::addressof(disarm_func));
    }
#else
    static_cast<void>(fence_seq);
    static_cast<void>(guest_addr);
    static_cast<void>(size);
    static_cast<void>(reason);
    static_cast<void>(disarm_func);
#endif
}

inline void DisarmLabelReadWatch(FenceSeq fence_seq, VAddr guest_addr, u64 size,
                                 SemanticWatchCancelReason reason = SemanticWatchCancelReason::ExplicitCancel) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        DisarmLabelReadWatchEnabled(fence_seq, guest_addr, size, reason, nullptr, nullptr);
    }
#else
    static_cast<void>(fence_seq);
    static_cast<void>(guest_addr);
    static_cast<void>(size);
    static_cast<void>(reason);
#endif
}

template <typename Func>
inline void HandleWriteFaultOnWatchedPage(VAddr addr, u64 size, u32 thread_id, VAddr rip,
                                         Func&& disarm_func) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        auto cb = [](void* ud, VAddr a, u64 s) noexcept {
            (*static_cast<std::remove_reference_t<Func>*>(ud))(a, s);
        };
        HandleWriteFaultOnWatchedPageEnabled(addr, size, thread_id, rip, cb, std::addressof(disarm_func));
    }
#else
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(thread_id);
    static_cast<void>(rip);
    static_cast<void>(disarm_func);
#endif
}

template <typename Func>
inline bool CheckCpuReadObservation(VAddr addr, u64 size, u32 thread_id, VAddr rip, Func&& disarm_func) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        auto cb = [](void* ud, VAddr a, u64 s) noexcept {
            (*static_cast<std::remove_reference_t<Func>*>(ud))(a, s);
        };
        return CheckCpuReadObservationEnabled(addr, size, thread_id, rip, cb, std::addressof(disarm_func));
    }
#else
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(thread_id);
    static_cast<void>(rip);
    static_cast<void>(disarm_func);
#endif
    return false;
}

template <typename Func>
inline bool CheckCpuReadObservation(VAddr addr, u64 size, u32 thread_id, Func&& disarm_func) noexcept {
    return CheckCpuReadObservation(addr, size, thread_id, 0, std::forward<Func>(disarm_func));
}

inline bool CheckCpuReadObservation(VAddr addr, u64 size, u32 thread_id, VAddr rip = 0) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? CheckCpuReadObservationEnabled(addr, size, thread_id, rip, nullptr, nullptr) : false;
#else
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(thread_id);
    static_cast<void>(rip);
    return false;
#endif
}

inline void RegisterCmdBufferSubmit(CmdBufferSeq cmd_buf, SubmitSeq submit_seq) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RegisterCmdBufferSubmitEnabled(cmd_buf, submit_seq);
    }
#else
    static_cast<void>(cmd_buf);
    static_cast<void>(submit_seq);
#endif
}

inline SubmitSeq LookupCmdBufferSubmit(CmdBufferSeq cmd_buf) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? LookupCmdBufferSubmitEnabled(cmd_buf) : 0;
#else
    static_cast<void>(cmd_buf);
    return 0;
#endif
}

inline void RegisterPendingReadbackForSubmit(ReadbackSeq readback_seq, CmdBufferSeq cmd_buf) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RegisterPendingReadbackForSubmitEnabled(readback_seq, cmd_buf);
    }
#else
    static_cast<void>(readback_seq);
    static_cast<void>(cmd_buf);
#endif
}

inline void PromotePendingReadbacksOnSubmit(CmdBufferSeq cmd_buf, SubmitSeq submit_seq,
                                          u64 signal_tick) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        PromotePendingReadbacksOnSubmitEnabled(cmd_buf, submit_seq, signal_tick);
    }
#else
    static_cast<void>(cmd_buf);
    static_cast<void>(submit_seq);
    static_cast<void>(signal_tick);
#endif
}

inline FenceSeq MatchFenceForWait(WaitSeq wait_seq, PacketSeq packet_seq, FrameSeq frame_seq,
                                  u32 queue_id, Pm4Engine engine, VAddr wait_addr, u32 ref,
                                  u32 mask, u32 function, PacketSeq prev_pkt, u32 prev_op) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        return MatchFenceForWaitEnabled(wait_seq, packet_seq, frame_seq, queue_id, engine,
                                        wait_addr, ref, mask, function, prev_pkt, prev_op);
    }
#else
    static_cast<void>(wait_seq);
    static_cast<void>(packet_seq);
    static_cast<void>(frame_seq);
    static_cast<void>(queue_id);
    static_cast<void>(engine);
    static_cast<void>(wait_addr);
    static_cast<void>(ref);
    static_cast<void>(mask);
    static_cast<void>(function);
    static_cast<void>(prev_pkt);
    static_cast<void>(prev_op);
#endif
    return 0;
}

inline void ArmConsumerProbe(FenceSeq fence_seq, WaitSeq wait_seq, PacketSeq wait_pkt,
                             ConsumerProbeKind probe_kind = ConsumerProbeKind::MatchedFence) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        ArmConsumerProbeEnabled(fence_seq, wait_seq, wait_pkt, probe_kind);
    }
#else
    static_cast<void>(fence_seq);
    static_cast<void>(wait_seq);
    static_cast<void>(wait_pkt);
    static_cast<void>(probe_kind);
#endif
}

inline void CheckConsumerOverlap(ProducerSeq consumer_prod, PacketSeq pkt, ProducerClass consumer_type,
                                 VAddr addr, u64 size, ConsumerAccessPath access_path = ConsumerAccessPath::Unknown,
                                 ResourceSeq consumer_res_id = 0, ResourceVersion consumer_res_ver = 0,
                                 ConsumerConfidence confidence = ConsumerConfidence::ExactResourceAndVersion,
                                 u64 pipeline_hash = 0, u64 shader_hash = 0) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        CheckConsumerOverlapEnabled(consumer_prod, pkt, consumer_type, addr, size, access_path,
                                    consumer_res_id, consumer_res_ver, confidence, pipeline_hash, shader_hash);
    }
#else
    static_cast<void>(consumer_prod);
    static_cast<void>(pkt);
    static_cast<void>(consumer_type);
    static_cast<void>(addr);
    static_cast<void>(size);
    static_cast<void>(access_path);
    static_cast<void>(consumer_res_id);
    static_cast<void>(consumer_res_ver);
    static_cast<void>(confidence);
    static_cast<void>(pipeline_hash);
    static_cast<void>(shader_hash);
#endif
}

inline void CheckConsumerOverlap(ProducerSeq consumer_prod, PacketSeq pkt, ProducerClass consumer_type,
                                 VAddr addr, u64 size, u32 access_kind, ResourceSeq consumer_res_id) noexcept {
    const ConsumerAccessPath path = (access_kind == 0) ? ConsumerAccessPath::ColorAttachment
                                                       : ConsumerAccessPath::StorageImage;
    CheckConsumerOverlap(consumer_prod, pkt, consumer_type, addr, size, path, consumer_res_id);
}

inline void RegisterSubmitTick(u64 signal_tick, SubmitSeq submit_seq) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RegisterSubmitTickEnabled(signal_tick, submit_seq);
    }
#else
    static_cast<void>(signal_tick);
    static_cast<void>(submit_seq);
#endif
}

inline SubmitSeq LookupSubmitSeq(u64 signal_tick) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? LookupSubmitSeqEnabled(signal_tick) : 0;
#else
    static_cast<void>(signal_tick);
    return 0;
#endif
}

inline void AdvanceConsumerProbes(PacketSeq current_packet, bool is_present) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        AdvanceConsumerProbesEnabled(current_packet, is_present);
    }
#else
    static_cast<void>(current_packet);
    static_cast<void>(is_present);
#endif
}

inline GuestMemoryWriteOrigin CurrentMemoryWriteOrigin() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    return Enabled() ? CurrentMemoryWriteOriginEnabled() : GuestMemoryWriteOrigin::GuestCpu;
#else
    return GuestMemoryWriteOrigin::GuestCpu;
#endif
}

inline void Add(Counter counter, u64 value = 1) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        AddEnabled(counter, value);
    }
#else
    static_cast<void>(counter);
    static_cast<void>(value);
#endif
}

inline void Subtract(Counter counter, u64 value = 1) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        SubtractEnabled(counter, value);
    }
#else
    static_cast<void>(counter);
    static_cast<void>(value);
#endif
}

inline void ObserveMax(Counter counter, u64 value) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        ObserveMaxEnabled(counter, value);
    }
#else
    static_cast<void>(counter);
    static_cast<void>(value);
#endif
}

inline void Record(EventType type, u64 arg0 = 0, u64 arg1 = 0) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordEnabled(type, arg0, arg1);
    }
#else
    static_cast<void>(type);
    static_cast<void>(arg0);
    static_cast<void>(arg1);
#endif
}

inline void CountPm4Packet(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                           uintptr_t address, u32 words, u32 header) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        CountPm4PacketEnabled(engine, queue_id, opcode, depth, address, words, header);
    }
#else
    static_cast<void>(engine);
    static_cast<void>(queue_id);
    static_cast<void>(opcode);
    static_cast<void>(depth);
    static_cast<void>(address);
    static_cast<void>(words);
    static_cast<void>(header);
#endif
}

class ScopedDuration {
public:
    explicit ScopedDuration(Counter counter_, EventType type_ = EventType::None,
                            u64 arg0_ = 0) noexcept
        : counter{counter_}, type{type_}, arg0{arg0_}, start_ns{Enabled() ? Timestamp() : 0} {}

    explicit ScopedDuration(bool enabled, Counter counter_, EventType type_ = EventType::None,
                            u64 arg0_ = 0) noexcept
        : counter{counter_}, type{type_}, arg0{arg0_}, start_ns{enabled ? Timestamp() : 0} {}

    ~ScopedDuration() {
        if (start_ns != 0) [[unlikely]] {
            RecordDurationEnabled(counter, type, start_ns, arg0);
        }
    }

private:
    Counter counter;
    EventType type;
    u64 arg0;
    u64 start_ns;
};

template <TimerSite Site>
class SampledDuration {
public:
    explicit SampledDuration(bool enabled = Enabled(),
                             u32 stage_ = InvalidTimerStage) noexcept
        : stage{stage_} {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        if (!enabled) {
            return;
        }
        constexpr u32 period = TimerSamplePeriod(Site);
        static_assert(std::has_single_bit(period));
        const u32 ticket = ++sequence;
        const u32 mixed = ticket * 0x9E3779B9U;
        if (mixed < (u32{1} << (32 - std::countr_zero(period)))) {
            start_ns = Timestamp();
        }
#else
        static_cast<void>(enabled);
#endif
    }

    ~SampledDuration() {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        if (start_ns != 0) [[unlikely]] {
            RecordTimerSampleEnabled(Site, start_ns, stage);
        }
#endif
    }

private:
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    inline static thread_local u32 sequence =
        (static_cast<u32>(Site) + 1U) * 0x85EBCA6BU;
    u64 start_ns{};
#endif
    u32 stage;
};

inline void RecordFrameSample(u32 frame_id, u64 present_start_ns) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (Enabled()) [[unlikely]] {
        RecordFrameSampleEnabled(frame_id, present_start_ns);
    }
#else
    static_cast<void>(frame_id);
    static_cast<void>(present_start_ns);
#endif
}

/// Writes the in-memory snapshot on a normal GUI shutdown. Returns an empty path when logging is
/// disabled or no telemetry was collected.
[[nodiscard]] std::filesystem::path Dump();

} // namespace Common::PerformanceTelemetry
