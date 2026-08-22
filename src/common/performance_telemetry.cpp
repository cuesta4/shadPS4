// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <zstd.h>

#include "common/path_util.h"
#include "common/performance_telemetry.h"
#include "common/thread.h"

namespace Common::PerformanceTelemetry {
namespace {

constexpr u64 RingCapacity = 8192_KB;
constexpr u64 RingMask = RingCapacity - 1;
static_assert(std::has_single_bit(RingCapacity));

constexpr size_t HistogramSubdivisions = 8;
constexpr size_t HistogramExactValues = 8;
constexpr size_t HistogramBucketCount =
    HistogramExactValues + (std::numeric_limits<u64>::digits - 3) * HistogramSubdivisions;

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
constexpr size_t Pm4EngineCount = 3;
constexpr size_t Pm4OpcodeCount = 256;
constexpr size_t Pm4WordOverflowBucket = 256;
constexpr size_t Pm4WordBucketCount = Pm4WordOverflowBucket + 1;
constexpr size_t Pm4RegisterSpaceCount = 4;
constexpr size_t Pm4RegisterCount = 0x1000;
constexpr size_t Pm4ControlCapacity = 1048576;
constexpr size_t Pm4WaitCapacity = 262144;
constexpr size_t Pm4HashProbeLimit = 32;
constexpr size_t TimerSiteCount = static_cast<size_t>(TimerSite::Count);
constexpr size_t TrackedStageCount = 6;
constexpr size_t StageReasonCount = 64;
constexpr size_t DynamicReasonCount = 16;
constexpr size_t DynamicGroupCount = 32;
constexpr size_t DescriptorReasonCount = 512;
constexpr size_t ImageFindPathCount = static_cast<size_t>(ImageFindPath::Count);
constexpr size_t StagingSiteCount = static_cast<size_t>(StagingSite::Count);
constexpr size_t StagingSourceCount = static_cast<size_t>(StagingSource::Count);
constexpr size_t StagingMemoryKindCount = static_cast<size_t>(StagingMemoryKind::Count);
constexpr size_t SubmitReasonCount = static_cast<size_t>(SubmitReason::Count);
constexpr size_t WritebackTriggerCount = static_cast<size_t>(WritebackTrigger::Count);
constexpr size_t StagingSizeBucketCount = std::numeric_limits<u64>::digits + 1;
constexpr size_t WritebackRecordCapacity = 4194304;
constexpr size_t FrameRecordCapacity = 524288;
constexpr size_t SyncPm4PacketCapacity = 4194304;
constexpr size_t ProducerRecordCapacity = 4194304;
constexpr size_t ProducerEndRecordCapacity = 4194304;
constexpr size_t ProducerFullRecordCapacity = 4194304;
constexpr size_t ResourceWriteRecordCapacity = 4194304;
constexpr size_t FenceRecordCapacity = 2097152;
constexpr size_t FenceEpochLinkCapacity = 4194304;
constexpr size_t FenceMatchAttemptCapacity = 2097152;
constexpr size_t FenceMatchDiagnosticCapacity = 1048576;
constexpr size_t ResourceEpochPromotedCapacity = 1048576;
constexpr size_t FenceResourceLinkCapacity = 2097152;
constexpr size_t WaitRecordCapacity = 2097152;
constexpr size_t WaitCompleteCapacity = 2097152;
constexpr size_t FirstConsumerCapacity = 1048576;
constexpr size_t FenceClassificationCapacity = 2097152;
constexpr size_t CpuAccessRecordCapacity = 2097152;
constexpr size_t CpuLabelAccessCapacity = 1048576;
constexpr size_t CpuMaterializationRecordCapacity = 1048576;
constexpr size_t StaleGuestAttemptRecordCapacity = 1048576;
constexpr size_t GpuAliasRecordCapacity = 1048576;
constexpr size_t ReadbackScheduleCapacity = 4194304;
constexpr size_t ReadbackSubmitCapacity = 4194304;
constexpr size_t ReadbackReadyCapacity = 4194304;
constexpr size_t ReadbackCommitCapacity = 4194304;
constexpr size_t ReadbackSourceTerminalCapacity = 1048576;
constexpr size_t GuestSourceConsumeCapacity = 1048576;
constexpr size_t ResourceLineageCapacity = 1048576;
constexpr size_t CpuReadObservationCapacity = 262144;
constexpr size_t ResourceBarrierLinkCapacity = 1048576;
constexpr size_t AcquireMemCapacity = 1048576;
constexpr size_t FenceSignalCapacity = 2097152;
constexpr size_t HostWaitCapacity = 4194304;
constexpr size_t SubmitRecordCapacity = 1048576;
constexpr size_t ShadowFencePolicyCapacity = 1048576;
constexpr size_t TraceGapCapacity = 131072;
constexpr size_t RingHealthCapacity = 2048;
constexpr size_t SemanticReadFaultCapacity = 524288;
constexpr size_t SemanticReadUnknownCapacity = 131072;
constexpr size_t SemanticWatchCancelCapacity = 524288;
constexpr size_t SemanticPageConflictWriteCapacity = 131072;
constexpr size_t FastpathCandidateCapacity = 65536;
constexpr size_t GpuAuthorityCreateCapacity = 65536;
constexpr size_t VirtualFenceCreateCapacity = 65536;
constexpr size_t VirtualWaitConsumeCapacity = 65536;
constexpr size_t AsyncLabelSignalCapacity = 65536;
constexpr size_t AuthorityGpuConsumeCapacity = 65536;
constexpr size_t AuthorityBarrierValidationCapacity = 65536;
constexpr size_t AuthorityRamDemandCapacity = 65536;
constexpr size_t LazyMaterializeBeginCapacity = 65536;
constexpr size_t LazyMaterializeEndCapacity = 65536;
constexpr size_t AuthorityRamConsumeCapacity = 65536;
constexpr size_t AuthorityCpuReadCapacity = 65536;
constexpr size_t AuthoritySupersedeCapacity = 65536;
constexpr size_t FastpathFallbackCapacity = 65536;
constexpr size_t ConservativeDownloadDecisionCapacity = 65536;
constexpr size_t AuthorityConservativeReadbackSuppressedCapacity = 65536;
constexpr size_t AuthorityHostMaterializeRequiredCapacity = 65536;
constexpr size_t FastpathWaitDecisionCapacity = 65536;
constexpr size_t VirtualFenceForcedCompletionCapacity = 65536;
constexpr size_t CpuToGpuLabelWaitCapacity = 65536;
constexpr size_t StageReasonBitCount = 6;
constexpr size_t DynamicReasonBitCount = 4;
constexpr size_t DynamicGroupBitCount = 5;
constexpr size_t DescriptorReasonBitCount = 9;
constexpr size_t StageFrameTimerCount = 8;
static_assert(static_cast<size_t>(TimerSite::StageSlowPath) + 1 == StageFrameTimerCount);
constexpr u32 StageReasonSamplePeriod = 16;
constexpr u32 DynamicReasonSamplePeriod = 16;
constexpr u32 DescriptorReasonSamplePeriod = 64;
constexpr u32 DescriptorCrossPipelineSamplePeriod = 64;
constexpr u32 ImageFindPathSamplePeriod = 32;
constexpr u32 StagingDetailSamplePeriod = 16;
static_assert(std::has_single_bit(WritebackRecordCapacity));
static_assert(std::has_single_bit(FrameRecordCapacity));
static_assert(std::has_single_bit(SyncPm4PacketCapacity));
static_assert(std::has_single_bit(ProducerRecordCapacity));
static_assert(std::has_single_bit(ProducerFullRecordCapacity));
static_assert(std::has_single_bit(ResourceWriteRecordCapacity));
static_assert(std::has_single_bit(FenceRecordCapacity));
static_assert(std::has_single_bit(FenceEpochLinkCapacity));
static_assert(std::has_single_bit(FenceMatchAttemptCapacity));
static_assert(std::has_single_bit(FenceMatchDiagnosticCapacity));
static_assert(std::has_single_bit(ResourceEpochPromotedCapacity));
static_assert(std::has_single_bit(FenceResourceLinkCapacity));
static_assert(std::has_single_bit(WaitRecordCapacity));
static_assert(std::has_single_bit(WaitCompleteCapacity));
static_assert(std::has_single_bit(FirstConsumerCapacity));
static_assert(std::has_single_bit(FenceClassificationCapacity));
static_assert(std::has_single_bit(CpuAccessRecordCapacity));
static_assert(std::has_single_bit(CpuLabelAccessCapacity));
static_assert(std::has_single_bit(CpuMaterializationRecordCapacity));
static_assert(std::has_single_bit(StaleGuestAttemptRecordCapacity));
static_assert(std::has_single_bit(GpuAliasRecordCapacity));
static_assert(std::has_single_bit(ReadbackScheduleCapacity));
static_assert(std::has_single_bit(ReadbackSubmitCapacity));
static_assert(std::has_single_bit(ReadbackReadyCapacity));
static_assert(std::has_single_bit(ReadbackCommitCapacity));
static_assert(std::has_single_bit(ReadbackSourceTerminalCapacity));
static_assert(std::has_single_bit(GuestSourceConsumeCapacity));
static_assert(std::has_single_bit(ResourceLineageCapacity));
static_assert(std::has_single_bit(CpuReadObservationCapacity));
static_assert(std::has_single_bit(ResourceBarrierLinkCapacity));
static_assert(std::has_single_bit(AcquireMemCapacity));
static_assert(std::has_single_bit(FenceSignalCapacity));
static_assert(std::has_single_bit(HostWaitCapacity));
static_assert(std::has_single_bit(SubmitRecordCapacity));
static_assert(std::has_single_bit(ShadowFencePolicyCapacity));
static_assert(std::has_single_bit(TraceGapCapacity));
static_assert(std::has_single_bit(RingHealthCapacity));
static_assert(std::has_single_bit(SemanticReadFaultCapacity));
static_assert(std::has_single_bit(SemanticReadUnknownCapacity));
static_assert(std::has_single_bit(SemanticWatchCancelCapacity));
static_assert(std::has_single_bit(SemanticPageConflictWriteCapacity));
static_assert(std::has_single_bit(FastpathCandidateCapacity));
static_assert(std::has_single_bit(GpuAuthorityCreateCapacity));
static_assert(std::has_single_bit(VirtualFenceCreateCapacity));
static_assert(std::has_single_bit(VirtualWaitConsumeCapacity));
static_assert(std::has_single_bit(AsyncLabelSignalCapacity));
static_assert(std::has_single_bit(AuthorityGpuConsumeCapacity));
static_assert(std::has_single_bit(AuthorityBarrierValidationCapacity));
static_assert(std::has_single_bit(AuthorityRamDemandCapacity));
static_assert(std::has_single_bit(LazyMaterializeBeginCapacity));
static_assert(std::has_single_bit(LazyMaterializeEndCapacity));
static_assert(std::has_single_bit(AuthorityRamConsumeCapacity));
static_assert(std::has_single_bit(AuthorityCpuReadCapacity));
static_assert(std::has_single_bit(AuthoritySupersedeCapacity));
static_assert(std::has_single_bit(FastpathFallbackCapacity));
static_assert(std::has_single_bit(ConservativeDownloadDecisionCapacity));
static_assert(std::has_single_bit(AuthorityConservativeReadbackSuppressedCapacity));
static_assert(std::has_single_bit(AuthorityHostMaterializeRequiredCapacity));
static_assert(std::has_single_bit(FastpathWaitDecisionCapacity));
static_assert(std::has_single_bit(VirtualFenceForcedCompletionCapacity));
static_assert(std::has_single_bit(CpuToGpuLabelWaitCapacity));

constexpr std::array TimerNames{
    "stage_refresh",          "stage_flat_copy",       "stage_srt_walker",
    "stage_resolve",          "stage_current_match",   "stage_search",
    "stage_specialization",   "stage_slow_path",       "dynamic_total",
    "dynamic_viewport",       "dynamic_depth_stencil", "dynamic_primitive",
    "dynamic_rasterization",  "dynamic_blend",         "descriptor_prepare",
    "descriptor_compare",     "descriptor_emit",       "render_prepare",
    "render_image_desc",      "render_state_build",    "image_find",
    "image_update",           "render_target_prepare", "scheduler_begin_rendering",
    "memory_notify",          "event_query_stores",    "event_query_notify",
    "staging_stream_single",
    "staging_stream_batch",   "staging_stream_slice",  "staging_image",
    "staging_upload_copies",  "staging_write_data",
    "staging_batch_deduplicate", "staging_batch_layout", "staging_batch_copy",
    "staging_batch_results",  "staging_batch_request_prepare", "staging_sparse_copy",
    "staging_sparse_shared_total", "staging_sparse_lock", "staging_sparse_plan_lookup",
    "staging_sparse_plan_build",
    "staging_sparse_payload_copy", "staging_sparse_finish",
    "staging_sparse_dense_lookup", "staging_sparse_span_refill",
    "staging_sparse_cold_path", "staging_stream_reuse",
    "descriptor_user_data",  "descriptor_buffers",
    "descriptor_textures",    "descriptor_capture",
};
static_assert(TimerNames.size() == TimerSiteCount);

constexpr std::array StageNames{
    "fragment", "tessellation_control", "tessellation_evaluation",
    "vertex",   "geometry",             "compute",
};
static_assert(StageNames.size() == TrackedStageCount);

constexpr std::array StageReasonBitNames{
    "stage_srt_walker",          "stage_tessellation_control",
    "stage_tessellation_eval",   "stage_descriptor_outside_user_data",
    "stage_fetch_pointer_outside_user_data", "stage_fetch_shader_unavailable",
};
static_assert(StageReasonBitNames.size() == StageReasonBitCount);

constexpr std::array DynamicReasonBitNames{
    "dynamic_generation", "dynamic_pipeline", "dynamic_indexed", "dynamic_feedback_loop",
};
static_assert(DynamicReasonBitNames.size() == DynamicReasonBitCount);

constexpr std::array DynamicGroupBitNames{
    "dynamic_viewport", "dynamic_depth_stencil", "dynamic_primitive",
    "dynamic_rasterization", "dynamic_blend_write_feedback",
};
static_assert(DynamicGroupBitNames.size() == DynamicGroupBitCount);

constexpr std::array DescriptorReasonBitNames{
    "descriptor_invalid_state", "descriptor_pipeline",      "descriptor_command_buffer",
    "descriptor_push_epoch",    "descriptor_uncacheable",   "descriptor_missing_cached",
    "descriptor_metadata",      "descriptor_buffer_info",   "descriptor_image_info",
};
static_assert(DescriptorReasonBitNames.size() == DescriptorReasonBitCount);

constexpr std::array ImageFindPathNames{
    "exact_cache",
    "perfect_scan",
    "overlap",
    "created",
};
static_assert(ImageFindPathNames.size() == ImageFindPathCount);

constexpr std::array StagingSiteNames{
    "stream_single", "stream_batch", "stream_slice",
    "image",         "upload_copies", "write_data",
};
static_assert(StagingSiteNames.size() == StagingSiteCount);

constexpr std::array StagingSourceNames{"guest", "host", "zero"};
static_assert(StagingSourceNames.size() == StagingSourceCount);

constexpr std::array StagingMemoryKindNames{"current_stream", "host_direct"};
static_assert(StagingMemoryKindNames.size() == StagingMemoryKindCount);

constexpr std::array SubmitReasonNames{
    "generic",          "guest_submit",       "writeback_eos", "writeback_eop",
    "writeback_release", "present_frame_build", "present_submit", "queue_present",
    "finish",           "wait_progress",
};
static_assert(SubmitReasonNames.size() == SubmitReasonCount);

constexpr std::array WritebackTriggerNames{
    "guest_submit", "event_write_eos", "event_write_eop", "release_mem",
    "garbage_collection",
};
static_assert(WritebackTriggerNames.size() == WritebackTriggerCount);

constexpr std::array ImageWriterNames{
    "none", "graphics_draw", "compute_dispatch", "compute_hle", "transfer",
    "cpu_upload",
};
static_assert(ImageWriterNames.size() == static_cast<size_t>(ImageWriter::Count));

constexpr std::array ProducerClassNames{
    "graphics_draw", "compute_dispatch", "compute_hle", "clear",
    "copy",          "resolve",          "color_target", "depth_target",
    "storage_image", "storage_buffer",   "dma",          "cpu",
};
static_assert(ProducerClassNames.size() == static_cast<size_t>(ProducerClass::Count));

constexpr std::array ResourceTypeNames{
    "image", "buffer", "color_target", "depth_target", "download_buffer",
};
static_assert(ResourceTypeNames.size() == static_cast<size_t>(ResourceType::Count));

constexpr std::array ResourceWriteKindNames{
    "storage_image", "storage_buffer", "color_target", "depth_target",
    "transfer",      "resolve",        "clear",        "dma",
    "cpu",
};
static_assert(ResourceWriteKindNames.size() == static_cast<size_t>(ResourceWriteKind::Count));

constexpr std::array FenceKindNames{
    "event_write_eos", "event_write_eop", "release_mem", "write_data",
};
static_assert(FenceKindNames.size() == static_cast<size_t>(FenceKind::Count));

constexpr std::array FenceStageScopeNames{
    "cs", "ps", "pipe", "all",
};
static_assert(FenceStageScopeNames.size() == static_cast<size_t>(FenceStageScope::Count));

constexpr std::array FenceClassificationNames{
    "gpu_only", "cpu_visible", "ambiguous",
};
static_assert(FenceClassificationNames.size() == static_cast<size_t>(FenceClassification::Count));

constexpr std::array FenceLinkReasonNames{
    "candidate_write_epoch", "direct_epoch_ref", "overlap_heuristic", "structural_shadow",
    "same_queue_prior_producer", "same_stage_prior_producer", "explicit_cache_scope",
    "global_conservative",
};
static_assert(FenceLinkReasonNames.size() == static_cast<size_t>(FenceLinkReason::Count));

constexpr std::array WaitConfidenceNames{
    "none", "low", "medium", "high", "exact",
};
static_assert(WaitConfidenceNames.size() == static_cast<size_t>(WaitConfidence::Count));

constexpr std::array CandidateRemoveReasonNames{
    "none", "matched_and_consumed", "superseded_by_new_fence", "explicit_write_to_label",
    "cpu_write_to_label", "pm4_write_data", "pm4_dma", "queue_reset",
    "command_buffer_end", "submission_boundary", "frame_boundary",
    "capacity_eviction", "invalidated", "unknown",
};
static_assert(CandidateRemoveReasonNames.size() == static_cast<size_t>(CandidateRemoveReason::Count));

constexpr std::array ConsumerProbeKindNames{
    "matched_fence", "structural_shadow_fence",
};
static_assert(ConsumerProbeKindNames.size() == static_cast<size_t>(ConsumerProbeKind::Count));

constexpr std::array ConsumerAccessPathNames{
    "unknown", "sampled_image", "storage_image", "uniform_buffer", "storage_buffer",
    "texel_buffer", "vertex_buffer", "index_buffer", "draw_indirect_args",
    "draw_indirect_count", "dispatch_indirect_args", "color_attachment",
    "depth_attachment", "image_alias_resolve", "buffer_from_image",
    "guest_ram_upload", "cpu_fault_read",
};
static_assert(ConsumerAccessPathNames.size() == static_cast<size_t>(ConsumerAccessPath::Count));

constexpr std::array ConsumerConfidenceNames{
    "exact_resource_and_version", "exact_resource", "exact_guest_range",
    "partial_guest_overlap", "alias_base_match", "heuristic",
};
static_assert(ConsumerConfidenceNames.size() == static_cast<size_t>(ConsumerConfidence::Count));

constexpr std::array ShadowBasisNames{
    "none", "adjacent_same_label", "queue_order_same_label", "heuristic",
};
static_assert(ShadowBasisNames.size() == static_cast<size_t>(ShadowBasis::Count));

constexpr std::array ShadowOwnerNames{
    "unknown", "guest_ram", "buffer", "image", "cpu",
};
static_assert(ShadowOwnerNames.size() == static_cast<size_t>(ShadowOwner::Count));

constexpr std::array PromotionReasonNames{
    "fence_readback_source", "fence_first_consumer_source", "cpu_observation",
    "alias_observation", "stale_guest_attempt", "diagnostic_trigger",
};
static_assert(PromotionReasonNames.size() == static_cast<size_t>(PromotionReason::Count));

constexpr std::array MemoryWriteOriginNames{
    "guest_cpu", "fence_signal", "readback_commit", "pm4_write_data",
    "pm4_dma", "video_out", "emulator_internal", "unknown",
};
static_assert(MemoryWriteOriginNames.size() == static_cast<size_t>(MemoryWriteOrigin::Count));
constexpr auto& GuestMemoryWriteOriginNames = MemoryWriteOriginNames;

constexpr std::array CpuAccessTypeNames{
    "read", "write",
};
static_assert(CpuAccessTypeNames.size() == static_cast<size_t>(CpuAccessType::Count));

constexpr std::array FallbackActionNames{
    "allowed", "forced_host_readback", "gpu_image_to_buffer",
    "gpu_image_to_image", "gpu_buffer_to_image", "unknown",
};
static_assert(FallbackActionNames.size() == static_cast<size_t>(FallbackAction::Count));

constexpr std::array AliasCopyKindNames{
    "image_to_image", "image_to_buffer", "buffer_to_image", "resolve",
    "retile", "format_reinterpret", "compute_conversion",
};
static_assert(AliasCopyKindNames.size() == static_cast<size_t>(AliasCopyKind::Count));

constexpr std::array SourceWatchStateNames{
    "active",
    "gpu_read",
    "gpu_alias_materialization",
    "guest_ram_gpu_upload",
    "cpu_read",
    "overwritten",
    "destroyed_or_unmapped",
    "session_end_unknown",
};
static_assert(SourceWatchStateNames.size() == static_cast<size_t>(SourceWatchState::Count));
constexpr auto& TerminalKindNames = SourceWatchStateNames;

constexpr std::array HostVersionOriginNames{
    "guest_cpu",
    "readback_commit",
    "pm4_write_data",
    "pm4_dma",
    "fence_signal",
    "unknown",
};
static_assert(HostVersionOriginNames.size() == static_cast<size_t>(HostVersionOrigin::Count));

constexpr std::array GuestSourceConsumePathNames{
    "stream_buffer_copy",
    "staging_buffer_copy",
    "buffer_upload",
    "image_upload",
};
static_assert(GuestSourceConsumePathNames.size() == static_cast<size_t>(GuestSourceConsumePath::Count));

constexpr std::array ResourceLineageKindNames{
    "gpu_to_gpu",
    "cpu_to_gpu",
};
static_assert(ResourceLineageKindNames.size() == static_cast<size_t>(ResourceLineageKind::Count));

constexpr std::array ReadWatchKindNames{
    "label",
    "data",
};
static_assert(ReadWatchKindNames.size() == static_cast<size_t>(ReadWatchKind::Count));

constexpr std::array ReadbackReasonNames{
    "fence_conservative", "cpu_read_fault", "preemptive_hot_range",
    "alias_fallback", "explicit_1x1", "debug_validation",
};
static_assert(ReadbackReasonNames.size() == static_cast<size_t>(ReadbackReason::Count));

constexpr std::array HostWaitReasonNames{
    "unknown", "scheduler_finish", "fence_cpu_visibility", "wait_reg_mem_progress",
    "stream_buffer_reuse", "resource_destruction", "present", "debug",
};
static_assert(HostWaitReasonNames.size() == static_cast<size_t>(HostWaitReason::Count));

constexpr std::array CorrelationStatusNames{
    "complete", "partial", "missing_context", "context_evicted",
    "not_applicable", "unsupported", "internal_error",
};
static_assert(CorrelationStatusNames.size() == static_cast<size_t>(CorrelationStatus::Count));

constexpr std::array FenceMatchFailureNames{
    "none",
    "no_candidate",
    "no_active_generation",
    "address_mismatch",
    "reference_mismatch",
    "mask_mismatch",
    "function_mismatch",
    "generation_superseded",
    "candidate_already_consumed",
    "queue_rejected",
    "stage_rejected",
    "unsupported_comparison",
    "unsupported_fence_kind",
    "state_missing",
    "internal_collision",
};
static_assert(FenceMatchFailureNames.size() == static_cast<size_t>(FenceMatchFailure::Count));

constexpr std::array RepresentationTransitionNames{
    "image_to_image", "image_to_buffer", "image_to_indirect", "image_to_cpu",
    "buffer_to_image", "buffer_to_buffer", "same_resource", "unknown", "not_observed",
};
static_assert(RepresentationTransitionNames.size() == static_cast<size_t>(RepresentationTransition::Count));

constexpr std::array TraceGapReasonNames{
    "ring_overwrite", "sampling", "filter", "writer_backpressure", "capture_disabled", "schema_unsupported",
};
static_assert(TraceGapReasonNames.size() == static_cast<size_t>(TraceGapReason::Count));

constexpr std::array TraceCaptureProfileNames{
    "sync_perf", "sync_semantic", "sync_fastpath_validation",
};
static_assert(TraceCaptureProfileNames.size() == static_cast<size_t>(TraceCaptureProfile::Count));

constexpr std::array SemanticReadOriginNames{
    "none",
    "guest_direct",
    "guest_hle",
    "wait_reg_mem_poll",
    "gpu_upload_from_guest_ram",
    "renderer_internal",
    "memory_tracker_internal",
    "telemetry_internal",
    "unknown_host",
};
static_assert(SemanticReadOriginNames.size() == static_cast<size_t>(SemanticReadOrigin::Count));

constexpr std::array SemanticWatchCancelReasonNames{
    "wait_completed",
    "generation_superseded",
    "observed",
    "overwritten",
    "page_conflict_write",
    "unmap",
    "explicit_cancel",
    "capture_shutdown",
    "livelock_break",
};
static_assert(SemanticWatchCancelReasonNames.size() == static_cast<size_t>(SemanticWatchCancelReason::Count));

struct StagingDetail {
    std::atomic<u64> allocations{};
    std::atomic<u64> bytes{};
    std::array<std::atomic<u64>, StagingSizeBucketCount> sizes{};
    std::array<std::atomic<u64>, StagingSourceCount> source_records{};
    std::array<std::atomic<u64>, StagingSourceCount> source_bytes{};
};

struct StagingMemoryTypeDetail {
    std::atomic<u32> memory_type{};
    std::atomic<u32> memory_heap{};
    std::atomic<u32> property_flags{};
    std::atomic_bool valid{};
};

struct SubmitTimingDetail {
    std::atomic<u64> calls{};
    std::atomic<u64> mutex_wait_ns{};
    std::atomic<u64> prepare_ns{};
    std::atomic<u64> driver_ns{};
    std::atomic<u64> post_ns{};
    std::atomic<u64> mutex_hold_ns{};
};

struct SubmitTimingSnapshot {
    u64 calls{};
    u64 mutex_wait_ns{};
    u64 prepare_ns{};
    u64 driver_ns{};
    u64 post_ns{};
    u64 mutex_hold_ns{};
};

struct FrameDetailSnapshot {
    std::array<std::array<u64, StageFrameTimerCount>, TrackedStageCount> stage_timer_ns{};
    std::array<std::array<u64, StageFrameTimerCount>, TrackedStageCount> stage_timer_samples{};
    std::array<std::array<u64, StageReasonBitCount>, TrackedStageCount> stage_reason_bits{};
    std::array<u64, DynamicReasonBitCount> dynamic_reason_bits{};
    std::array<u64, DynamicGroupBitCount> dynamic_pending_group_bits{};
    std::array<u64, DynamicGroupBitCount> dynamic_emitted_group_bits{};
    std::array<u64, DescriptorReasonBitCount> descriptor_reason_bits{};
    std::array<u64, ImageFindPathCount> image_find_paths{};
    std::array<u64, StagingSiteCount> staging_allocations{};
    std::array<u64, StagingSiteCount> staging_bytes{};
    std::array<std::array<u64, StagingSourceCount>, StagingSiteCount> staging_source_records{};
    std::array<std::array<u64, StagingSourceCount>, StagingSiteCount> staging_source_bytes{};
    std::array<SubmitTimingSnapshot, SubmitReasonCount> submits{};
};

struct WritebackRecord {
    std::atomic<u64> committed_sequence{};
    u64 timestamp_ns{};
    WritebackImageSample sample{};
};

struct WritebackSnapshot {
    u64 sequence{};
    u64 timestamp_ns{};
    WritebackImageSample sample{};
};

constexpr std::array FrameCounters{
    Counter::Pm4Packets,
    Counter::GfxSubmits,
    Counter::AscSubmits,
    Counter::GcpActiveNs,
    Counter::GcpBlockedNs,
    Counter::Draws,
    Counter::Dispatches,
    Counter::DrawCpuNs,
    Counter::DispatchCpuNs,
    Counter::PipelineHits,
    Counter::PipelineMisses,
    Counter::ShaderModuleCompileNs,
    Counter::ShaderModuleCompileJobs,
    Counter::ShaderModulePendingDraws,
    Counter::ShaderModuleQueueWaitNs,
    Counter::RenderTargetHits,
    Counter::RenderTargetMisses,
    Counter::ImageTokenHits,
    Counter::ImageTokenMisses,
    Counter::DescriptorHits,
    Counter::DescriptorMisses,
    Counter::StagingBytes,
    Counter::BarrierCalls,
    Counter::CopyCalls,
    Counter::CopyBytes,
    Counter::StageCacheCurrentHits,
    Counter::StageCacheSearchHits,
    Counter::StageCacheMisses,
    Counter::StageCacheUncacheable,
    Counter::StageSpecializationBuilds,
    Counter::DynamicStateHits,
    Counter::DynamicStateMisses,
    Counter::DynamicStateEmptyCommits,
    Counter::DynamicStateInvalidations,
    Counter::DriverSubmitCalls,
    Counter::DriverSubmitNs,
    Counter::SubmitMutexWaitNs,
    Counter::SubmitMutexHoldNs,
    Counter::SubmitPrepareNs,
    Counter::SubmitPostNs,
    Counter::DriverPresentCalls,
    Counter::DriverPresentNs,
    Counter::PresentMutexWaitNs,
    Counter::PresentMutexHoldNs,
    Counter::WritebackCalls,
    Counter::WritebackBytes,
    Counter::WritebackBatches,
    Counter::WritebackFenceDeferrals,
    Counter::WritebackFlushes,
    Counter::WritebackDrainCalls,
    Counter::WritebackDrainEmpty,
    Counter::WritebackCandidates,
    Counter::WritebackSameEpoch,
    Counter::MemoryNotifyCalls,
    Counter::MemoryNotifyNoWatchCalls,
    Counter::MemoryNotifyActiveWatchCalls,
    Counter::EventQueryCalls,
    Counter::EventQueryCounterPairs,
    Counter::EventQueryNoWatchCalls,
    Counter::EventQueryActiveWatchCalls,
    Counter::EventQueryMatchedWatchCalls,
    Counter::RenderColorAttachments,
    Counter::RenderDepthAttachments,
    Counter::RenderTargetRebinds,
    Counter::GpuIdleGaps,
    Counter::GpuIdleGapNs,
    Counter::StageSlowCurrentHits,
    Counter::StageSlowOtherHits,
    Counter::StageSlowSearchComparisons,
    Counter::StagingBatchSamples,
    Counter::StagingBatchRequests,
    Counter::StagingBatchCanonicalCopies,
    Counter::StagingBatchGuestCopies,
    Counter::StagingBatchExactReuses,
    Counter::StagingBatchSubrangeReuses,
    Counter::StagingBatchRequestedBytes,
    Counter::StagingBatchCanonicalBytes,
    Counter::StagingBatchAllocatedBytes,
    Counter::StagingSparseCopySamples,
    Counter::StagingSparseCopyRequests,
    Counter::StagingSparsePlanHits,
    Counter::StagingSparsePlanMisses,
    Counter::StagingSparsePlansBuilt,
    Counter::StagingSparseCopyReferenceFallbacks,
    Counter::StagingSparseMappedRuns,
    Counter::StagingSparseZeroRuns,
    Counter::StagingSparseRequestedBytes,
    Counter::StagingSparseCopiedBytes,
    Counter::StagingSparseReferenceBytes,
    Counter::StagingSparseMappedBytes,
    Counter::StagingSparseZeroBytes,
    Counter::StagingSparsePlanMissEmpty,
    Counter::StagingSparsePlanMissGeneration,
    Counter::StagingSparsePlanMissConflict,
    Counter::StagingSparsePlanMissUncacheable,
    Counter::StagingSparseNonTemporalRuns,
    Counter::StagingSparseNonTemporalBytes,
    Counter::StagingSparseNonTemporalFences,
    Counter::StagingSparseMergeablePairs,
    Counter::StagingSparseMergeableBytes,
    Counter::DescriptorCrossPipelineSamples,
    Counter::DescriptorCrossPipelineExactStateHits,
    Counter::DescriptorCrossPipelineCompatibleLayoutHits,
    Counter::DescriptorCrossPipelineReusableHits,
    Counter::StagingSparseDenseHotHits,
    Counter::StagingSparseDenseCacheHits,
    Counter::StagingSparseDenseRefills,
    Counter::StagingSparseDenseFallbackRequests,
    Counter::StagingSparseDenseHotBytes,
    Counter::StagingSparseDenseCacheBytes,
    Counter::StagingSparseDenseFallbackBytes,
    Counter::StagingCurrentStreamBatches,
    Counter::StagingCurrentStreamBytes,
    Counter::StagingHostDirectBatches,
    Counter::StagingHostDirectBytes,
    Counter::StagingGpuPromotedBatches,
    Counter::StagingGpuPromotedBytes,
    Counter::StagingReuseCandidates,
    Counter::StagingReuseHits,
    Counter::StagingReuseMismatches,
    Counter::StagingReuseCooldownSkips,
    Counter::StagingReuseInvalidations,
    Counter::StagingReuseWarmups,
    Counter::StagingReuseCandidateBytes,
    Counter::StagingReuseAvoidedBytes,
    Counter::StagingReuseWarmupBytes,
    Counter::WritebackFlushesAvoided,
    Counter::GpuFenceWaitBypasses,
    Counter::GpuFenceWaitBarriers,
    Counter::GpuFenceWaitForcedFlushes,
    Counter::GpuFenceTokenOverflows,
    Counter::AcquireMemCalls,
    Counter::AcquireMemBarriers,
    Counter::EventWriteFlushCalls,
    Counter::EventWriteFlushBarriers,
    Counter::RenderTargetSyncBarriers,
    Counter::RenderTargetSampledAsTexture,
    Counter::RenderTargetTransitions,
    Counter::BruteForceBarriers,
    Counter::BruteForceHostFinishes,
};

struct FrameRecord {
    u64 sequence{};
    u64 timestamp_ns{};
    u64 interval_ns{};
    u64 present_duration_ns{};
    u32 frame_id{};
    std::array<u64, FrameCounters.size()> counters{};
    std::array<u64, TimerSiteCount> timer_ns{};
    std::array<u64, TimerSiteCount> timer_samples{};
    FrameDetailSnapshot details{};
};
static_assert(std::has_single_bit(Pm4ControlCapacity));
static_assert(std::has_single_bit(Pm4WaitCapacity));
constexpr std::array<u32, Pm4RegisterSpaceCount> Pm4RegisterOpcodes{0x68, 0x69, 0x76, 0x79};

struct Pm4OpcodeDetail {
    std::atomic<u64> packets{};
    std::atomic<u64> words{};
    std::atomic<u64> predicated{};
    std::atomic<u64> shader_compute{};
    std::atomic<u64> max_depth{};
    std::array<std::atomic<u64>, Pm4WordBucketCount> word_counts{};
};

struct Pm4RegisterDetail {
    std::atomic<u64> packets{};
    std::atomic<u64> words{};
    std::atomic<u64> changed{};
};

struct Pm4ControlDetail {
    std::atomic<u64> hash{};
    u64 identity{};
    u32 control0{};
    u32 control1{};
    std::atomic<u64> packets{};
};

struct Pm4WaitDetail {
    std::atomic<u64> hash{};
    u64 location{};
    u32 identity{};
    u32 control{};
    u32 reference{};
    u32 mask{};
    u32 poll_interval{};
    std::atomic<u64> packets{};
    std::atomic<u64> failed_tests{};
    std::atomic<u64> immediate_passes{};
    std::atomic<u64> vo_sleeps{};
};

struct Pm4Detail {
    std::array<std::array<Pm4OpcodeDetail, Pm4OpcodeCount>, Pm4EngineCount> opcodes{};
    std::array<std::array<std::array<Pm4RegisterDetail, Pm4RegisterCount>,
                          Pm4RegisterSpaceCount>,
               Pm4EngineCount>
        registers{};
    std::array<Pm4ControlDetail, Pm4ControlCapacity> controls{};
    std::array<Pm4WaitDetail, Pm4WaitCapacity> waits{};
    std::atomic<u64> register_overflow{};
    std::atomic<u64> control_overflow{};
    std::atomic<u64> wait_overflow{};
};
#endif

constexpr std::array CounterNames{
    "pm4_packets",          "pm4_type2_packets",  "dcb_bytes",
    "ccb_bytes",            "acb_bytes",          "gfx_submits",
    "asc_submits",          "gcp_wakes",          "queue_scans",
    "queue_resumes",        "queue_front_loads",  "gcp_active_ns",
    "gcp_blocked_ns",
    "queue_ready_ns",       "queue_resume_ns",    "ib_depth_max",
    "draws",
    "dispatches",           "draw_cpu_ns",        "dispatch_cpu_ns",
    "pipeline_hits",        "pipeline_misses",    "pipeline_compile_ns",
    "render_target_hits",   "render_target_misses", "image_token_hits",
    "image_token_misses",   "descriptor_hits",    "descriptor_misses",
    "buffer_token_hits",    "buffer_token_misses", "stream_slice_hits",
    "stream_slice_misses",  "staging_bytes",      "barrier_calls",
    "copy_calls",           "copy_bytes",         "timeline_polls",
    "timeline_poll_ns",     "pending_op_empty_hits", "pending_op_known_tick_hits",
    "pending_op_refreshes", "stage_cache_current_hits", "stage_cache_search_hits",
    "stage_cache_misses",   "stage_cache_uncacheable", "stage_fingerprint_collisions",
    "stage_specialization_builds", "stage_program_creates", "stage_permutation_compiles",
    "stage_permutation_hits", "fetch_shader_cache_hits", "fetch_shader_cache_misses",
    "fetch_shader_words",   "dynamic_state_hits", "dynamic_state_misses",
    "dynamic_state_empty_commits",
    "driver_submit_calls",  "driver_submit_ns",
    "driver_present_calls", "driver_present_ns",  "submit_queue_depth_max",
    "wait_calls",           "wait_ns",            "writeback_calls",
    "writeback_bytes",      "writeback_ns",       "writeback_enqueue_ns",
    "writeback_batches",    "writeback_stale_skips", "writeback_fence_deferrals",
    "writeback_flushes",    "present_prepare_ns",
    "present_cpu_ns",       "gpu_idle_gaps",       "gpu_idle_gap_ns",
    "memory_watch_arms",    "memory_watch_cancels", "memory_watch_wakeups",
    "memory_watch_arm_failures", "memory_watch_fallbacks", "memory_notify_calls",
    "memory_notify_pages",
    "memory_notify_tracked_pages", "memory_notify_callbacks", "memory_notify_ns",
    "memory_notify_cpu",    "memory_notify_command_processor",
    "memory_notify_gpu_completion", "memory_notify_map", "memory_notify_unmap",
    "dynamic_state_invalidations", "render_color_attachments", "render_depth_attachments",
    "render_target_rebinds", "submit_mutex_wait_ns", "submit_mutex_hold_ns",
    "submit_prepare_ns", "submit_post_ns", "present_mutex_wait_ns",
    "present_mutex_hold_ns", "memory_notify_no_watch_calls",
    "memory_notify_active_watch_calls", "event_query_calls", "event_query_counter_pairs",
    "event_query_no_watch_calls", "event_query_active_watch_calls",
    "event_query_matched_watch_calls", "writeback_drain_calls", "writeback_drain_empty",
    "writeback_candidates", "writeback_same_epoch",
    "stage_slow_current_hits", "stage_slow_other_hits", "stage_slow_search_comparisons",
    "staging_batch_samples", "staging_batch_requests", "staging_batch_canonical_copies",
    "staging_batch_guest_copies", "staging_batch_exact_reuses",
    "staging_batch_subrange_reuses", "staging_batch_requested_bytes",
    "staging_batch_canonical_bytes", "staging_batch_allocated_bytes",
    "staging_sparse_copy_samples", "staging_sparse_copy_requests",
    "staging_sparse_plan_hits", "staging_sparse_plan_misses",
    "staging_sparse_plans_built", "staging_sparse_copy_reference_fallbacks",
    "staging_sparse_mapped_runs", "staging_sparse_zero_runs",
    "staging_sparse_requested_bytes", "staging_sparse_copied_bytes",
    "staging_sparse_reference_bytes", "staging_sparse_mapped_bytes",
    "staging_sparse_zero_bytes", "staging_sparse_plan_miss_empty",
    "staging_sparse_plan_miss_generation", "staging_sparse_plan_miss_conflict",
    "staging_sparse_plan_miss_uncacheable", "staging_sparse_non_temporal_runs",
    "staging_sparse_non_temporal_bytes", "staging_sparse_non_temporal_fences",
    "staging_sparse_mergeable_pairs", "staging_sparse_mergeable_bytes",
    "descriptor_cross_pipeline_samples", "descriptor_cross_pipeline_exact_state_hits",
    "descriptor_cross_pipeline_compatible_layout_hits",
    "descriptor_cross_pipeline_reusable_hits",
    "staging_sparse_dense_hot_hits", "staging_sparse_dense_cache_hits",
    "staging_sparse_dense_refills", "staging_sparse_dense_fallback_requests",
    "staging_sparse_dense_hot_bytes", "staging_sparse_dense_cache_bytes",
    "staging_sparse_dense_fallback_bytes", "staging_current_stream_batches",
    "staging_current_stream_bytes", "staging_host_direct_batches",
    "staging_host_direct_bytes", "staging_gpu_promoted_batches",
    "staging_gpu_promoted_bytes", "staging_reuse_candidates", "staging_reuse_hits",
    "staging_reuse_mismatches", "staging_reuse_cooldown_skips",
    "staging_reuse_invalidations", "staging_reuse_warmups",
    "staging_reuse_candidate_bytes", "staging_reuse_avoided_bytes",
    "staging_reuse_warmup_bytes", "writeback_flushes_avoided",
    "gpu_fence_wait_bypasses", "gpu_fence_wait_barriers",
    "gpu_fence_wait_forced_flushes", "gpu_fence_token_overflows",
    "acquire_mem_calls", "acquire_mem_barriers",
    "event_write_flush_calls", "event_write_flush_barriers",
    "render_target_sync_barriers",
    "render_target_sampled_as_texture",
    "render_target_transitions",
    "brute_force_barriers",
    "brute_force_host_finishes",
    "wait_reg_mem_calls",
    "wait_reg_mem_spin_ns",
    "wait_reg_mem_spins",
    "priority_ops_wait_ns",
    "priority_ops_drain_count",
    "priority_ops_execute_ns",
    "fence_total",
    "fence_gpu_only",
    "fence_cpu_visible",
    "fence_ambiguous",
    "fence_epoch_links",
    "wait_matched_fences",
    "wait_unmatched_fences",
    "cpu_accesses",
    "cpu_stale_accesses",
    "cpu_materializations",
    "stale_guest_attempts",
    "gpu_alias_materializations",
    "readback_schedule_count",
    "readback_submit_count",
    "readback_ready_count",
    "readback_commit_count",
    "fence_signal_count",
    "host_wait_count",
    "host_wait_finish_count",
    "host_wait_cpu_visibility_count",
    "host_wait_wait_progress_count",
    "fence_match_attempt_count",
    "fence_match_success_count",
    "fence_match_failure_count",
    "wait_complete_count",
    "first_consumer_count",
    "cpu_label_access_count",
    "trace_gap_count",
    "ring_health_count",
    "fence_match_diagnostic_count",
    "resource_epoch_promoted_count",
    "fence_resource_link_count",
    "shadow_consumer_probe_hits",
    "matched_consumer_probe_hits",
    "semantic_label_watch_arms",
    "semantic_data_watch_arms",
    "semantic_label_watch_pages",
    "semantic_data_watch_pages",
    "semantic_read_faults",
    "cpu_read_observation_count",
    "semantic_read_faults_total",
    "semantic_read_guest_direct",
    "semantic_read_guest_hle",
    "semantic_read_wait_reg_mem_poll",
    "semantic_read_gpu_upload",
    "semantic_read_renderer_internal",
    "semantic_read_memory_tracker_internal",
    "semantic_read_telemetry_internal",
    "semantic_read_unknown_host",
    "semantic_label_guest_reads",
    "semantic_data_guest_reads",
    "semantic_precision_unknown_reads",
    "semantic_active_label_watches",
    "semantic_active_data_watches",
    "semantic_page_conflict_writes",
    "semantic_watch_cancel_wait_complete",
    "semantic_watch_cancel_overwritten",
    "semantic_watch_cancel_page_conflict",
    "semantic_fault_livelock_breaks",
    "fastpath_candidates",
    "fastpath_taken",
    "fastpath_rejected_signature",
    "fastpath_rejected_resource",
    "fastpath_rejected_queue",
    "fastpath_rejected_overlap",
    "fastpath_rejected_lifetime",
    "eager_3k_downloads",
    "lazy_3k_materializations",
    "authority_created",
    "authority_gpu_first_consumer",
    "authority_ram_first_consumer",
    "authority_cpu_read",
    "authority_superseded_without_host_use",
    "authority_destroyed_without_host_use",
    "virtual_wait_consumed",
    "physical_wait_fallback",
    "async_label_writes",
    "stale_label_callbacks",
    "barrier_validation_success",
    "barrier_validation_failure",
    "ram_demand_events",
    "ram_demand_unique_authorities",
    "materialize_success",
    "materialize_failure",
    "materialize_byte_mismatch",
    "conservative_download_considered",
    "conservative_download_suppressed",
    "conservative_3k_readbacks",
    "fastpath_wait_progress_submits",
    "virtual_fence_forced_completions",
    "virtual_wait_fallback",
    "virtual_fence_retired",
    "shader_module_compile_ns",
    "shader_module_compile_jobs",
    "shader_module_pending_draws",
    "shader_module_queue_wait_ns",
    "shader_module_queue_depth_max",
};
static_assert(CounterNames.size() == static_cast<size_t>(Counter::Count));

constexpr std::array ConservativeDownloadDecisionNames{
    "legacy_download",
    "suppress_authority",
    "materialize_required",
    "generation_mismatch_fallback",
};
static_assert(ConservativeDownloadDecisionNames.size() == static_cast<size_t>(ConservativeDownloadDecision::Count));

constexpr std::array FastpathWaitDecisionNames{
    "virtualized",
    "legacy",
    "forced_complete",
};
static_assert(FastpathWaitDecisionNames.size() == static_cast<size_t>(FastpathWaitDecision::Count));

constexpr std::array VirtualFenceForcedCompletionReasonNames{
    "host_side_effect",
    "ram_demand",
    "cpu_label_visibility",
    "irq",
    "queue_transition",
    "unmap",
    "fallback",
};
static_assert(VirtualFenceForcedCompletionReasonNames.size() == static_cast<size_t>(VirtualFenceForcedCompletionReason::Count));

constexpr std::array FastpathEligibilityNames{
    "eligible",
    "rejected",
};
static_assert(FastpathEligibilityNames.size() == static_cast<size_t>(FastpathEligibility::Count));

constexpr std::array FastpathRejectReasonNames{
    "none",
    "title_mismatch",
    "producer_not_compute",
    "resource_not_storage",
    "size_not_3072",
    "tiled_unsupported",
    "eos_invalid_command",
    "eos_invalid_value",
    "eos_has_irq",
    "eos_size_mismatch",
    "wait_mismatch",
    "acquire_mismatch",
    "queue_order_mismatch",
    "lifetime_invalid",
    "overlap_conflict",
};
static_assert(FastpathRejectReasonNames.size() == static_cast<size_t>(FastpathRejectReason::Count));

constexpr std::array VirtualWaitResultNames{
    "virtualized",
    "physical_wait",
    "fallback",
};
static_assert(VirtualWaitResultNames.size() == static_cast<size_t>(VirtualWaitResult::Count));

constexpr std::array AsyncLabelActionNames{
    "wrote",
    "stale_generation_skipped",
    "already_satisfied",
};
static_assert(AsyncLabelActionNames.size() == static_cast<size_t>(AsyncLabelAction::Count));

constexpr std::array LazyMaterializeResultNames{
    "success",
    "image_generation_mismatch",
    "unmapped",
    "download_failed",
    "fallback",
};
static_assert(LazyMaterializeResultNames.size() == static_cast<size_t>(LazyMaterializeResult::Count));

constexpr std::array FastpathFallbackPhaseNames{
    "recognizer",
    "pending_download",
    "fence",
    "wait",
    "barrier",
    "ram_materialize",
    "cpu_fault",
    "lifetime",
    "overlap",
};
static_assert(FastpathFallbackPhaseNames.size() == static_cast<size_t>(FastpathFallbackPhase::Count));

constexpr std::array FastpathFallbackReasonNames{
    "none",
    "signature_mismatch",
    "resource_invalid",
    "queue_invalid",
    "overlap_conflict",
    "lifetime_mismatch",
    "barrier_failed",
    "download_failed",
    "unknown_origin",
};
static_assert(FastpathFallbackReasonNames.size() == static_cast<size_t>(FastpathFallbackReason::Count));

constexpr std::array EventNames{
    "none",
    "gcp_active",
    "gcp_blocked",
    "submit_done",
    "timeline_complete",
    "driver_submit",
    "wait",
    "writeback",
    "present_prepare",
    "present_cpu",
    "driver_present",
    "frame_presented",
    "gpu_idle_gap",
    "pm4_packet",
    "pm4_wait_begin",
    "pm4_wait_end",
    "pm4_acquire_mem",
    "pm4_release_mem",
    "pm4_event_write",
    "pm4_surface_sync",
    "vulkan_pipeline_barrier",
    "vulkan_image_layout",
    "vulkan_draw",
    "vulkan_dispatch",
    "vulkan_copy",
    "vulkan_submit",
    "memory_watch_arm",
    "memory_watch_wake",
    "alias_sync",
    "sync_pm4_packet",
    "producer_begin",
    "producer_end",
    "resource_write",
    "fence_create",
    "fence_epoch_link",
    "wait_create",
    "fence_classification",
    "cpu_memory_access",
    "cpu_read_requires_materialization",
    "stale_guest_source_attempt",
    "gpu_alias_materialize",
    "readback_schedule",
    "readback_submit",
    "readback_ready",
    "readback_commit",
    "fence_signal",
    "host_wait",
    "submit_record",
    "fence_match_attempt",
    "wait_complete",
    "first_consumer",
    "cpu_label_access",
    "trace_gap",
    "ring_health",
    "producer_record",
    "shadow_fence_policy",
    "fence_match_diagnostic",
    "resource_epoch_promoted",
    "fence_resource_link",
};
static_assert(EventNames.size() == static_cast<size_t>(EventType::Count));

constexpr std::array HistogramCounters{
    Counter::GcpActiveNs,      Counter::GcpBlockedNs,    Counter::QueueReadyNs,
    Counter::QueueResumeNs,    Counter::DrawCpuNs,       Counter::DispatchCpuNs,
    Counter::PipelineCompileNs, Counter::TimelinePollNs, Counter::DriverSubmitNs,
    Counter::DriverPresentNs,  Counter::WaitNs,          Counter::WritebackNs,
    Counter::WritebackEnqueueNs,
    Counter::PresentPrepareNs, Counter::PresentCpuNs,    Counter::GpuIdleGapNs,
    Counter::MemoryNotifyNs,
    Counter::SubmitMutexWaitNs,
    Counter::SubmitMutexHoldNs,
    Counter::SubmitPrepareNs,
    Counter::SubmitPostNs,
    Counter::PresentMutexWaitNs,
    Counter::PresentMutexHoldNs,
    Counter::SubmitQueueDepthMax,
    Counter::FetchShaderWords,
    Counter::ShaderModuleCompileNs,
    Counter::ShaderModuleQueueWaitNs,
    Counter::ShaderModuleQueueDepthMax,
};

constexpr std::array HistogramNames{
    "gcp_active_ns",       "gcp_blocked_ns",    "queue_ready_ns",
    "queue_resume_ns",     "draw_cpu_ns",       "dispatch_cpu_ns",
    "pipeline_compile_ns", "timeline_poll_ns",  "driver_submit_ns",
    "driver_present_ns",   "wait_ns",           "writeback_ns",
    "writeback_enqueue_ns",
    "present_prepare_ns",  "present_cpu_ns",    "gpu_idle_gap_ns",
    "memory_notify_ns",
    "submit_mutex_wait_ns",
    "submit_mutex_hold_ns",
    "submit_prepare_ns",
    "submit_post_ns",
    "present_mutex_wait_ns",
    "present_mutex_hold_ns",
    "submit_queue_depth",
    "fetch_shader_words",
    "shader_module_compile_ns",
    "shader_module_queue_wait_ns",
    "shader_module_queue_depth",
};
static_assert(HistogramCounters.size() == HistogramNames.size());

constexpr u8 NoHistogram = std::numeric_limits<u8>::max();
constexpr auto HistogramIndices = [] {
    std::array<u8, static_cast<size_t>(Counter::Count)> indices{};
    indices.fill(NoHistogram);
    for (u8 i = 0; i < HistogramCounters.size(); ++i) {
        indices[static_cast<size_t>(HistogramCounters[i])] = i;
    }
    return indices;
}();

struct EventSlot {
    std::atomic<u64> timestamp_ns{};
    std::atomic<u64> arg0{};
    std::atomic<u64> arg1{};
    std::atomic<u64> metadata{};
    std::atomic<u64> committed_sequence{};
};

struct ThreadRing {
    explicit ThreadRing(u32 id_, std::string name_) : id{id_}, name{std::move(name_)} {}

    alignas(64) std::atomic<u64> next_sequence{};
    std::array<EventSlot, RingCapacity> events{};
    alignas(64) std::array<std::atomic<u64>, static_cast<size_t>(Counter::Count)> counters{};
    std::array<std::atomic<u64>, 256> opcodes{};
    std::array<std::array<std::atomic<u64>, HistogramBucketCount>, HistogramCounters.size()>
        histograms{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    Pm4Detail pm4{};
    std::array<std::atomic<u64>, TimerSiteCount> timer_ns{};
    std::array<std::atomic<u64>, TimerSiteCount> timer_samples{};
    std::array<std::array<std::atomic<u64>, HistogramBucketCount>, TimerSiteCount>
        timer_histograms{};
    std::array<std::array<std::atomic<u64>, TimerSiteCount>, TrackedStageCount>
        stage_timer_ns{};
    std::array<std::array<std::atomic<u64>, TimerSiteCount>, TrackedStageCount>
        stage_timer_samples{};
    std::array<std::array<std::atomic<u64>, StageReasonCount>, TrackedStageCount>
        stage_reasons{};
    std::array<std::array<std::atomic<u64>, StageReasonBitCount>, TrackedStageCount>
        stage_reason_bits{};
    std::array<std::atomic<u64>, DynamicReasonCount> dynamic_reasons{};
    std::array<std::atomic<u64>, DynamicReasonBitCount> dynamic_reason_bits{};
    std::array<std::atomic<u64>, DynamicGroupCount> dynamic_pending_groups{};
    std::array<std::atomic<u64>, DynamicGroupCount> dynamic_emitted_groups{};
    std::array<std::atomic<u64>, DynamicGroupBitCount> dynamic_pending_group_bits{};
    std::array<std::atomic<u64>, DynamicGroupBitCount> dynamic_emitted_group_bits{};
    std::array<std::atomic<u64>, DescriptorReasonCount> descriptor_reasons{};
    std::array<std::atomic<u64>, DescriptorReasonBitCount> descriptor_reason_bits{};
    std::array<std::atomic<u64>, ImageFindPathCount> image_find_paths{};
    std::array<StagingDetail, StagingSiteCount> staging{};
    std::array<SubmitTimingDetail, SubmitReasonCount> submits{};
    std::array<std::array<std::atomic<u64>, 3>, WritebackTriggerCount> writeback_drains{};
#endif
    u32 id;
    std::string name;
};

struct EventSnapshot {
    u64 timestamp_ns;
    u64 arg0;
    u64 arg1;
    u64 sequence;
    u32 thread_id;
    EventType type;
};

std::mutex g_rings_mutex;
std::vector<std::unique_ptr<ThreadRing>> g_rings;
std::atomic_bool g_dumping{};
thread_local ThreadRing* g_thread_ring{};
const u64 g_session_start_ns = Timestamp();
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
template <typename Sample>
struct RelationalRecord {
    std::atomic<u64> committed_sequence{};
    u64 timestamp_ns{};
    Sample sample{};
};

std::atomic<EventSeq> g_event_sequence{};
std::atomic<PacketSeq> g_packet_sequence{};
std::atomic<ProducerSeq> g_producer_sequence{};
std::atomic<FenceSeq> g_fence_sequence{};
std::atomic<FenceGen> g_label_generation_counter{};
std::atomic<WaitSeq> g_wait_sequence{};
std::atomic<ResourceSeq> g_resource_sequence{};
std::atomic<ResourceVersion> g_resource_version{};
std::atomic<SubmitSeq> g_submit_sequence{};
std::atomic<CpuAccessSeq> g_cpu_access_sequence{};
std::atomic<ReadbackSeq> g_readback_sequence{};
std::atomic<CmdBufferSeq> g_cmdbuf_sequence{};

std::unique_ptr<std::array<RelationalRecord<SyncPm4PacketSample>, SyncPm4PacketCapacity>> g_sync_pm4_records;
std::unique_ptr<std::array<RelationalRecord<ProducerBeginSample>, ProducerRecordCapacity>> g_producer_begin_records;
std::unique_ptr<std::array<RelationalRecord<ProducerEndSample>, ProducerEndRecordCapacity>> g_producer_end_records;
std::unique_ptr<std::array<RelationalRecord<ProducerRecordSample>, ProducerFullRecordCapacity>> g_producer_full_records;
std::unique_ptr<std::array<RelationalRecord<ResourceWriteSample>, ResourceWriteRecordCapacity>> g_resource_write_records;
std::unique_ptr<std::array<RelationalRecord<FenceCreateSample>, FenceRecordCapacity>> g_fence_create_records;
std::unique_ptr<std::array<RelationalRecord<FenceEpochLinkSample>, FenceEpochLinkCapacity>> g_fence_epoch_link_records;
std::unique_ptr<std::array<RelationalRecord<FenceMatchAttemptSample>, FenceMatchAttemptCapacity>> g_fence_match_records;
std::unique_ptr<std::array<RelationalRecord<FenceMatchDiagnosticSample>, FenceMatchDiagnosticCapacity>> g_fence_match_diagnostic_records;
std::unique_ptr<std::array<RelationalRecord<ResourceEpochPromotedSample>, ResourceEpochPromotedCapacity>> g_resource_epoch_promoted_records;
std::unique_ptr<std::array<RelationalRecord<FenceResourceLinkSample>, FenceResourceLinkCapacity>> g_fence_resource_link_records;
std::unique_ptr<std::array<RelationalRecord<WaitCreateSample>, WaitRecordCapacity>> g_wait_create_records;
std::unique_ptr<std::array<RelationalRecord<WaitCompleteSample>, WaitCompleteCapacity>> g_wait_complete_records;
std::unique_ptr<std::array<RelationalRecord<FirstConsumerSample>, FirstConsumerCapacity>> g_first_consumer_records;
std::unique_ptr<std::array<RelationalRecord<FenceClassificationSample>, FenceClassificationCapacity>> g_fence_classification_records;
std::unique_ptr<std::array<RelationalRecord<CpuAccessSample>, CpuAccessRecordCapacity>> g_cpu_access_records;
std::unique_ptr<std::array<RelationalRecord<CpuLabelAccessSample>, CpuLabelAccessCapacity>> g_cpu_label_records;
std::unique_ptr<std::array<RelationalRecord<CpuMaterializationSample>, CpuMaterializationRecordCapacity>> g_cpu_materialization_records;
std::unique_ptr<std::array<RelationalRecord<StaleGuestAttemptSample>, StaleGuestAttemptRecordCapacity>> g_stale_guest_records;
std::unique_ptr<std::array<RelationalRecord<GpuAliasMaterializeSample>, GpuAliasRecordCapacity>> g_gpu_alias_records;
std::unique_ptr<std::array<RelationalRecord<ReadbackScheduleSample>, ReadbackScheduleCapacity>> g_readback_schedule_records;
std::unique_ptr<std::array<RelationalRecord<ReadbackSubmitSample>, ReadbackSubmitCapacity>> g_readback_submit_records;
std::unique_ptr<std::array<RelationalRecord<ReadbackReadySample>, ReadbackReadyCapacity>> g_readback_ready_records;
std::unique_ptr<std::array<RelationalRecord<ReadbackCommitSample>, ReadbackCommitCapacity>> g_readback_commit_records;
std::unique_ptr<std::array<RelationalRecord<ReadbackSourceTerminalSample>, ReadbackSourceTerminalCapacity>> g_readback_source_terminal_records;
std::unique_ptr<std::array<RelationalRecord<GuestSourceConsumeSample>, GuestSourceConsumeCapacity>> g_guest_source_consume_records;
std::unique_ptr<std::array<RelationalRecord<ResourceLineageSample>, ResourceLineageCapacity>> g_resource_lineage_records;
std::unique_ptr<std::array<RelationalRecord<CpuReadObservationSample>, CpuReadObservationCapacity>> g_cpu_read_observation_records;
std::unique_ptr<std::array<RelationalRecord<SemanticReadFaultSample>, SemanticReadFaultCapacity>> g_semantic_read_fault_records;
std::unique_ptr<std::array<RelationalRecord<SemanticReadUnknownSample>, SemanticReadUnknownCapacity>> g_semantic_read_unknown_records;
std::unique_ptr<std::array<RelationalRecord<SemanticWatchCancelSample>, SemanticWatchCancelCapacity>> g_semantic_watch_cancel_records;
std::unique_ptr<std::array<RelationalRecord<SemanticPageConflictWriteSample>, SemanticPageConflictWriteCapacity>> g_semantic_page_conflict_records;
std::unique_ptr<std::array<RelationalRecord<ResourceBarrierLinkSample>, ResourceBarrierLinkCapacity>> g_resource_barrier_link_records;
std::unique_ptr<std::array<RelationalRecord<AcquireMemSample>, AcquireMemCapacity>> g_acquire_mem_records;
std::unique_ptr<std::array<RelationalRecord<FenceSignalSample>, FenceSignalCapacity>> g_fence_signal_records;
std::unique_ptr<std::array<RelationalRecord<HostWaitSample>, HostWaitCapacity>> g_host_wait_records;
std::unique_ptr<std::array<RelationalRecord<SubmitRecordSample>, SubmitRecordCapacity>> g_submit_records;
std::unique_ptr<std::array<RelationalRecord<ShadowFencePolicySample>, ShadowFencePolicyCapacity>> g_shadow_fence_records;
std::unique_ptr<std::array<RelationalRecord<TraceGapSample>, TraceGapCapacity>> g_trace_gap_records;
std::unique_ptr<std::array<RelationalRecord<RingHealthSample>, RingHealthCapacity>> g_ring_health_records;
std::unique_ptr<std::array<RelationalRecord<FastpathCandidateSample>, FastpathCandidateCapacity>> g_fastpath_candidate_records;
std::unique_ptr<std::array<RelationalRecord<GpuAuthorityCreateSample>, GpuAuthorityCreateCapacity>> g_gpu_authority_create_records;
std::unique_ptr<std::array<RelationalRecord<VirtualFenceCreateSample>, VirtualFenceCreateCapacity>> g_virtual_fence_create_records;
std::unique_ptr<std::array<RelationalRecord<VirtualWaitConsumeSample>, VirtualWaitConsumeCapacity>> g_virtual_wait_consume_records;
std::unique_ptr<std::array<RelationalRecord<AsyncLabelSignalSample>, AsyncLabelSignalCapacity>> g_async_label_signal_records;
std::unique_ptr<std::array<RelationalRecord<AuthorityGpuConsumeSample>, AuthorityGpuConsumeCapacity>> g_authority_gpu_consume_records;
std::unique_ptr<std::array<RelationalRecord<AuthorityBarrierValidationSample>, AuthorityBarrierValidationCapacity>> g_authority_barrier_validation_records;
std::unique_ptr<std::array<RelationalRecord<AuthorityRamDemandSample>, AuthorityRamDemandCapacity>> g_authority_ram_demand_records;
std::unique_ptr<std::array<RelationalRecord<LazyMaterializeBeginSample>, LazyMaterializeBeginCapacity>> g_lazy_materialize_begin_records;
std::unique_ptr<std::array<RelationalRecord<LazyMaterializeEndSample>, LazyMaterializeEndCapacity>> g_lazy_materialize_end_records;
std::unique_ptr<std::array<RelationalRecord<AuthorityRamConsumeSample>, AuthorityRamConsumeCapacity>> g_authority_ram_consume_records;
std::unique_ptr<std::array<RelationalRecord<AuthorityCpuReadSample>, AuthorityCpuReadCapacity>> g_authority_cpu_read_records;
std::unique_ptr<std::array<RelationalRecord<AuthoritySupersedeSample>, AuthoritySupersedeCapacity>> g_authority_supersede_records;
std::unique_ptr<std::array<RelationalRecord<FastpathFallbackSample>, FastpathFallbackCapacity>> g_fastpath_fallback_records;
std::unique_ptr<std::array<RelationalRecord<ConservativeDownloadDecisionSample>, ConservativeDownloadDecisionCapacity>> g_conservative_download_decision_records;
std::unique_ptr<std::array<RelationalRecord<AuthorityConservativeReadbackSuppressedSample>, AuthorityConservativeReadbackSuppressedCapacity>> g_authority_conservative_readback_suppressed_records;
std::unique_ptr<std::array<RelationalRecord<AuthorityHostMaterializeRequiredSample>, AuthorityHostMaterializeRequiredCapacity>> g_authority_host_materialize_required_records;
std::unique_ptr<std::array<RelationalRecord<FastpathWaitDecisionSample>, FastpathWaitDecisionCapacity>> g_fastpath_wait_decision_records;
std::unique_ptr<std::array<RelationalRecord<VirtualFenceForcedCompletionSample>, VirtualFenceForcedCompletionCapacity>> g_virtual_fence_forced_completion_records;
std::unique_ptr<std::array<RelationalRecord<CpuToGpuLabelWaitSample>, CpuToGpuLabelWaitCapacity>> g_cpu_to_gpu_label_wait_records;
std::unique_ptr<std::array<WritebackRecord, WritebackRecordCapacity>> g_writeback_records;
std::unique_ptr<std::array<FrameRecord, FrameRecordCapacity>> g_frame_records;

std::atomic<u64> g_sync_pm4_written{};
std::atomic<u64> g_producer_begin_written{};
std::atomic<u64> g_producer_end_written{};
std::atomic<u64> g_producer_full_written{};
std::atomic<u64> g_resource_write_written{};
std::atomic<u64> g_fence_create_written{};
std::atomic<u64> g_fence_epoch_link_written{};
std::atomic<u64> g_fence_match_written{};
std::atomic<u64> g_fence_match_diagnostic_written{};
std::atomic<u64> g_resource_epoch_promoted_written{};
std::atomic<u64> g_fence_resource_link_written{};
std::atomic<u64> g_wait_create_written{};
std::atomic<u64> g_wait_complete_written{};
std::atomic<u64> g_first_consumer_written{};
std::atomic<u64> g_fence_classification_written{};
std::atomic<u64> g_cpu_access_written{};
std::atomic<u64> g_cpu_label_written{};
std::atomic<u64> g_cpu_materialization_written{};
std::atomic<u64> g_stale_guest_written{};
std::atomic<u64> g_gpu_alias_written{};
std::atomic<u64> g_readback_schedule_written{};
std::atomic<u64> g_readback_submit_written{};
std::atomic<u64> g_readback_ready_written{};
std::atomic<u64> g_readback_commit_written{};
std::atomic<u64> g_readback_source_terminal_written{};
std::atomic<u64> g_guest_source_consume_written{};
std::atomic<u64> g_resource_lineage_written{};
std::atomic<u64> g_cpu_read_observation_written{};
std::atomic<u64> g_semantic_read_fault_written{};
std::atomic<u64> g_semantic_read_unknown_written{};
std::atomic<u64> g_semantic_watch_cancel_written{};
std::atomic<u64> g_semantic_page_conflict_written{};
std::atomic<u64> g_resource_barrier_link_written{};
std::atomic<u64> g_acquire_mem_written{};
std::atomic<u64> g_fence_signal_written{};
std::atomic<u64> g_host_wait_written{};
std::atomic<u64> g_submit_written{};
std::atomic<u64> g_shadow_fence_written{};
std::atomic<u64> g_trace_gap_written{};
std::atomic<u64> g_ring_health_written{};
std::atomic<u64> g_fastpath_candidate_written{};
std::atomic<u64> g_gpu_authority_create_written{};
std::atomic<u64> g_virtual_fence_create_written{};
std::atomic<u64> g_virtual_wait_consume_written{};
std::atomic<u64> g_async_label_signal_written{};
std::atomic<u64> g_authority_gpu_consume_written{};
std::atomic<u64> g_authority_barrier_validation_written{};
std::atomic<u64> g_authority_ram_demand_written{};
std::atomic<u64> g_lazy_materialize_begin_written{};
std::atomic<u64> g_lazy_materialize_end_written{};
std::atomic<u64> g_authority_ram_consume_written{};
std::atomic<u64> g_authority_cpu_read_written{};
std::atomic<u64> g_authority_supersede_written{};
std::atomic<u64> g_fastpath_fallback_written{};
std::atomic<u64> g_conservative_download_decision_written{};
std::atomic<u64> g_authority_conservative_readback_suppressed_written{};
std::atomic<u64> g_authority_host_materialize_required_written{};
std::atomic<u64> g_fastpath_wait_decision_written{};
std::atomic<u64> g_virtual_fence_forced_completion_written{};
std::atomic<u64> g_cpu_to_gpu_label_wait_written{};
std::mutex g_guest_label_writes_mutex;
struct LastGuestWriteInfo {
    u64 timestamp_ns{0};
    u32 value{0};
    u64 thread_id{0};
};
std::unordered_map<VAddr, LastGuestWriteInfo> g_guest_label_writes;
std::atomic<u64> g_candidate_sequence{};
std::atomic<u64> g_authority_sequence{};
std::atomic<u64> g_virtual_fence_sequence{};
std::atomic<u64> g_ram_demand_sequence{};
std::atomic<u64> g_ram_demand_group_sequence{};
std::atomic<u64> g_materialize_sequence{};
std::atomic<u64> g_consumer_sequence{};
std::atomic<u64> g_writeback_sequence{};
std::atomic<u64> g_watch_seq{1};

std::mutex g_frame_mutex;
u64 g_frame_sequence{};
u64 g_previous_frame_timestamp{};
std::array<u64, FrameCounters.size()> g_previous_frame_counters{};
std::array<u64, TimerSiteCount> g_previous_frame_timer_ns{};
std::array<u64, TimerSiteCount> g_previous_frame_timer_samples{};
FrameDetailSnapshot g_previous_frame_details{};
std::array<StagingMemoryTypeDetail, StagingMemoryKindCount> g_staging_memory_types{};

struct LabelGenerationState {
    FenceSeq fence_seq{};
    FenceGen generation{};
    VAddr addr{};
    u64 value{};
    u64 mask{0xffffffff};
    PacketSeq create_packet{};
    PacketSeq last_packet{};
    u32 queue_id{};
    u32 gpu_wait_count{};
    u32 cpu_label_read_count{};
    u32 cpu_label_write_count{};
    u32 cpu_protected_data_read_count{};
    u32 evidence_bits{};
    bool irq_requested{};
    bool superseded{false};
    bool valid{false};
};

constexpr size_t ActiveLabelTableCapacity = 4096;
std::mutex g_active_labels_mutex;
std::array<LabelGenerationState, ActiveLabelTableCapacity> g_active_labels{};
std::array<FenceCandidateDebugState, ActiveLabelTableCapacity> g_fence_candidate_debug{};

constexpr size_t SubmitTickMapCapacity = 4096;
struct SubmitTickEntry {
    u64 signal_tick{0};
    SubmitSeq submit_seq{0};
};
std::mutex g_submit_tick_mutex;
std::array<SubmitTickEntry, SubmitTickMapCapacity> g_submit_tick_map{};

struct ActiveSourceWatchKey {
    ResourceId resource_id{0};
    ResourceVersion resource_version{0};

    bool operator==(const ActiveSourceWatchKey& other) const noexcept {
        return resource_id == other.resource_id && resource_version == other.resource_version;
    }
};

struct ActiveSourceWatchKeyHash {
    size_t operator()(const ActiveSourceWatchKey& k) const noexcept {
        return std::hash<u64>{}(k.resource_id ^ (k.resource_version * 0x9E3779B97F4A7C15ULL));
    }
};

std::mutex g_source_watches_mutex;
std::unordered_map<ActiveSourceWatchKey, ReadbackSourceWatch, ActiveSourceWatchKeyHash> g_active_source_watches;

struct HostVersionRange {
    VAddr start{};
    VAddr end{};
    HostVersionState state{};
};

std::mutex g_host_version_mutex;
std::vector<HostVersionRange> g_host_version_ranges;

std::mutex g_read_interests_mutex;
std::unordered_map<VAddr, std::vector<ReadWatchInterest>> g_read_watch_interests;

std::mutex g_cmdbuf_submits_mutex;
std::unordered_map<CmdBufferSeq, SubmitSeq> g_cmdbuf_to_submit;
std::unordered_map<CmdBufferSeq, std::vector<ReadbackSeq>> g_cmdbuf_pending_readbacks;

struct RangeCoherenceEntry {
    VAddr addr{};
    u64 size{};
    ResourceVersion latest_version{};
    ResourceType authoritative_owner{ResourceType::Image};
    ResourceSeq authoritative_resource{};
    ResourceVersion host_version{};
    u64 authoritative_tick{};
    ProducerSeq producer_seq{};
    bool valid{};
};

constexpr size_t RangeCoherenceCapacity = 16384;
std::mutex g_range_coherence_mutex;
std::array<RangeCoherenceEntry, RangeCoherenceCapacity> g_range_coherence{};

struct ProtectedRange {
    VAddr addr{};
    u64 size{};
    ResourceSeq resource_id{};
    ResourceVersion version{};
    ResourceType resource_type{};
};

struct ConsumerProbeEntry {
    FenceSeq fence_seq{};
    WaitSeq wait_seq{};
    PacketSeq wait_packet{};
    ConsumerProbeKind probe_kind{ConsumerProbeKind::MatchedFence};
    std::array<ProtectedRange, 8> ranges{};
    u32 range_count{0};
    u32 packets_seen{0};
    u32 producers_seen{0};
    bool resolved{false};
    bool valid{false};
};

constexpr size_t ConsumerProbeCapacity = 64;
std::mutex g_consumer_probe_mutex;
std::array<ConsumerProbeEntry, ConsumerProbeCapacity> g_consumer_probes{};

struct CurrentProducerContext {
    ProducerSeq seq{0};
    PacketSeq packet_seq{0};
    ProducerClass type{ProducerClass::GraphicsDraw};
    u32 queue_id{0};
    u64 start_ns{0};
    u32 write_range_count{0};
    u64 write_bytes{0};
    u32 write_resource_count{0};
    bool is_promoted{false};
    bool active{false};
};
inline thread_local CurrentProducerContext tl_producer{};

struct CurrentMemoryContext {
    GuestMemoryWriteOrigin origin{GuestMemoryWriteOrigin::GuestCpu};
    FenceSeq fence_seq{0};
};
inline thread_local CurrentMemoryContext tl_memory{};

[[nodiscard]] constexpr bool TestWaitCondition(u32 value, u32 function, u32 mask, u32 ref) noexcept {
    const u32 test_val = value & mask;
    const u32 test_ref = ref & mask;
    switch (function) {
    case 0: return true;
    case 1: return test_val < test_ref;
    case 2: return test_val <= test_ref;
    case 3: return test_val == test_ref;
    case 4: return test_val != test_ref;
    case 5: return test_val >= test_ref;
    case 6: return test_val > test_ref;
    default: return false;
    }
}

void EnsureRelationalStorageInitialized() {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] {
        g_sync_pm4_records = std::make_unique<std::array<RelationalRecord<SyncPm4PacketSample>, SyncPm4PacketCapacity>>();
        g_producer_begin_records = std::make_unique<std::array<RelationalRecord<ProducerBeginSample>, ProducerRecordCapacity>>();
        g_producer_end_records = std::make_unique<std::array<RelationalRecord<ProducerEndSample>, ProducerEndRecordCapacity>>();
        g_producer_full_records = std::make_unique<std::array<RelationalRecord<ProducerRecordSample>, ProducerFullRecordCapacity>>();
        g_resource_write_records = std::make_unique<std::array<RelationalRecord<ResourceWriteSample>, ResourceWriteRecordCapacity>>();
        g_fence_create_records = std::make_unique<std::array<RelationalRecord<FenceCreateSample>, FenceRecordCapacity>>();
        g_fence_epoch_link_records = std::make_unique<std::array<RelationalRecord<FenceEpochLinkSample>, FenceEpochLinkCapacity>>();
        g_fence_match_records = std::make_unique<std::array<RelationalRecord<FenceMatchAttemptSample>, FenceMatchAttemptCapacity>>();
        g_fence_match_diagnostic_records = std::make_unique<std::array<RelationalRecord<FenceMatchDiagnosticSample>, FenceMatchDiagnosticCapacity>>();
        g_resource_epoch_promoted_records = std::make_unique<std::array<RelationalRecord<ResourceEpochPromotedSample>, ResourceEpochPromotedCapacity>>();
        g_fence_resource_link_records = std::make_unique<std::array<RelationalRecord<FenceResourceLinkSample>, FenceResourceLinkCapacity>>();
        g_wait_create_records = std::make_unique<std::array<RelationalRecord<WaitCreateSample>, WaitRecordCapacity>>();
        g_wait_complete_records = std::make_unique<std::array<RelationalRecord<WaitCompleteSample>, WaitCompleteCapacity>>();
        g_first_consumer_records = std::make_unique<std::array<RelationalRecord<FirstConsumerSample>, FirstConsumerCapacity>>();
        g_fence_classification_records = std::make_unique<std::array<RelationalRecord<FenceClassificationSample>, FenceClassificationCapacity>>();
        g_cpu_access_records = std::make_unique<std::array<RelationalRecord<CpuAccessSample>, CpuAccessRecordCapacity>>();
        g_cpu_label_records = std::make_unique<std::array<RelationalRecord<CpuLabelAccessSample>, CpuLabelAccessCapacity>>();
        g_cpu_materialization_records = std::make_unique<std::array<RelationalRecord<CpuMaterializationSample>, CpuMaterializationRecordCapacity>>();
        g_stale_guest_records = std::make_unique<std::array<RelationalRecord<StaleGuestAttemptSample>, StaleGuestAttemptRecordCapacity>>();
        g_gpu_alias_records = std::make_unique<std::array<RelationalRecord<GpuAliasMaterializeSample>, GpuAliasRecordCapacity>>();
        g_readback_schedule_records = std::make_unique<std::array<RelationalRecord<ReadbackScheduleSample>, ReadbackScheduleCapacity>>();
        g_readback_submit_records = std::make_unique<std::array<RelationalRecord<ReadbackSubmitSample>, ReadbackSubmitCapacity>>();
        g_readback_ready_records = std::make_unique<std::array<RelationalRecord<ReadbackReadySample>, ReadbackReadyCapacity>>();
        g_readback_commit_records = std::make_unique<std::array<RelationalRecord<ReadbackCommitSample>, ReadbackCommitCapacity>>();
        g_readback_source_terminal_records = std::make_unique<std::array<RelationalRecord<ReadbackSourceTerminalSample>, ReadbackSourceTerminalCapacity>>();
        g_guest_source_consume_records = std::make_unique<std::array<RelationalRecord<GuestSourceConsumeSample>, GuestSourceConsumeCapacity>>();
        g_resource_lineage_records = std::make_unique<std::array<RelationalRecord<ResourceLineageSample>, ResourceLineageCapacity>>();
        g_cpu_read_observation_records = std::make_unique<std::array<RelationalRecord<CpuReadObservationSample>, CpuReadObservationCapacity>>();
        g_semantic_read_fault_records = std::make_unique<std::array<RelationalRecord<SemanticReadFaultSample>, SemanticReadFaultCapacity>>();
        g_semantic_read_unknown_records = std::make_unique<std::array<RelationalRecord<SemanticReadUnknownSample>, SemanticReadUnknownCapacity>>();
        g_semantic_watch_cancel_records = std::make_unique<std::array<RelationalRecord<SemanticWatchCancelSample>, SemanticWatchCancelCapacity>>();
        g_semantic_page_conflict_records = std::make_unique<std::array<RelationalRecord<SemanticPageConflictWriteSample>, SemanticPageConflictWriteCapacity>>();
        g_resource_barrier_link_records = std::make_unique<std::array<RelationalRecord<ResourceBarrierLinkSample>, ResourceBarrierLinkCapacity>>();
        g_acquire_mem_records = std::make_unique<std::array<RelationalRecord<AcquireMemSample>, AcquireMemCapacity>>();
        g_fence_signal_records = std::make_unique<std::array<RelationalRecord<FenceSignalSample>, FenceSignalCapacity>>();
        g_host_wait_records = std::make_unique<std::array<RelationalRecord<HostWaitSample>, HostWaitCapacity>>();
        g_submit_records = std::make_unique<std::array<RelationalRecord<SubmitRecordSample>, SubmitRecordCapacity>>();
        g_shadow_fence_records = std::make_unique<std::array<RelationalRecord<ShadowFencePolicySample>, ShadowFencePolicyCapacity>>();
        g_trace_gap_records = std::make_unique<std::array<RelationalRecord<TraceGapSample>, TraceGapCapacity>>();
        g_ring_health_records = std::make_unique<std::array<RelationalRecord<RingHealthSample>, RingHealthCapacity>>();
        g_fastpath_candidate_records = std::make_unique<std::array<RelationalRecord<FastpathCandidateSample>, FastpathCandidateCapacity>>();
        g_gpu_authority_create_records = std::make_unique<std::array<RelationalRecord<GpuAuthorityCreateSample>, GpuAuthorityCreateCapacity>>();
        g_virtual_fence_create_records = std::make_unique<std::array<RelationalRecord<VirtualFenceCreateSample>, VirtualFenceCreateCapacity>>();
        g_virtual_wait_consume_records = std::make_unique<std::array<RelationalRecord<VirtualWaitConsumeSample>, VirtualWaitConsumeCapacity>>();
        g_async_label_signal_records = std::make_unique<std::array<RelationalRecord<AsyncLabelSignalSample>, AsyncLabelSignalCapacity>>();
        g_authority_gpu_consume_records = std::make_unique<std::array<RelationalRecord<AuthorityGpuConsumeSample>, AuthorityGpuConsumeCapacity>>();
        g_authority_barrier_validation_records = std::make_unique<std::array<RelationalRecord<AuthorityBarrierValidationSample>, AuthorityBarrierValidationCapacity>>();
        g_authority_ram_demand_records = std::make_unique<std::array<RelationalRecord<AuthorityRamDemandSample>, AuthorityRamDemandCapacity>>();
        g_lazy_materialize_begin_records = std::make_unique<std::array<RelationalRecord<LazyMaterializeBeginSample>, LazyMaterializeBeginCapacity>>();
        g_lazy_materialize_end_records = std::make_unique<std::array<RelationalRecord<LazyMaterializeEndSample>, LazyMaterializeEndCapacity>>();
        g_authority_ram_consume_records = std::make_unique<std::array<RelationalRecord<AuthorityRamConsumeSample>, AuthorityRamConsumeCapacity>>();
        g_authority_cpu_read_records = std::make_unique<std::array<RelationalRecord<AuthorityCpuReadSample>, AuthorityCpuReadCapacity>>();
        g_authority_supersede_records = std::make_unique<std::array<RelationalRecord<AuthoritySupersedeSample>, AuthoritySupersedeCapacity>>();
        g_fastpath_fallback_records = std::make_unique<std::array<RelationalRecord<FastpathFallbackSample>, FastpathFallbackCapacity>>();
        g_conservative_download_decision_records = std::make_unique<std::array<RelationalRecord<ConservativeDownloadDecisionSample>, ConservativeDownloadDecisionCapacity>>();
        g_authority_conservative_readback_suppressed_records = std::make_unique<std::array<RelationalRecord<AuthorityConservativeReadbackSuppressedSample>, AuthorityConservativeReadbackSuppressedCapacity>>();
        g_authority_host_materialize_required_records = std::make_unique<std::array<RelationalRecord<AuthorityHostMaterializeRequiredSample>, AuthorityHostMaterializeRequiredCapacity>>();
        g_fastpath_wait_decision_records = std::make_unique<std::array<RelationalRecord<FastpathWaitDecisionSample>, FastpathWaitDecisionCapacity>>();
        g_virtual_fence_forced_completion_records = std::make_unique<std::array<RelationalRecord<VirtualFenceForcedCompletionSample>, VirtualFenceForcedCompletionCapacity>>();
        g_cpu_to_gpu_label_wait_records = std::make_unique<std::array<RelationalRecord<CpuToGpuLabelWaitSample>, CpuToGpuLabelWaitCapacity>>();
        g_writeback_records = std::make_unique<std::array<WritebackRecord, WritebackRecordCapacity>>();
        g_frame_records = std::make_unique<std::array<FrameRecord, FrameRecordCapacity>>();
    });
}

template <typename Sample, size_t Capacity>
void WriteRelationalSample(std::unique_ptr<std::array<RelationalRecord<Sample>, Capacity>>& storage,
                           std::atomic<u64>& written_counter,
                           const Sample& sample) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    EnsureRelationalStorageInitialized();
    if (!storage) {
        return;
    }
    const u64 sequence = written_counter.fetch_add(1, std::memory_order_relaxed);
    auto& slot = (*storage)[sequence & (Capacity - 1)];
    slot.timestamp_ns = Timestamp();
    slot.sample = sample;
    slot.committed_sequence.store(sequence + 1, std::memory_order_release);
}
#endif

[[nodiscard]] ThreadRing* GetThreadRing() {
    if (g_thread_ring != nullptr) {
        return g_thread_ring;
    }
    if (g_dumping.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    std::scoped_lock lock{g_rings_mutex};
    if (g_dumping.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    const u32 id = static_cast<u32>(g_rings.size());
    std::string name{Common::GetCurrentThreadNameView()};
    if (name.empty()) {
        name = "unnamed";
    }
    g_thread_ring =
        g_rings.emplace_back(std::make_unique<ThreadRing>(id, std::move(name))).get();
    return g_thread_ring;
}

void AddSingleWriter(std::atomic<u64>& counter, u64 value) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
}

void ObserveSingleWriterMax(std::atomic<u64>& counter, u64 value) noexcept {
    if (const u64 current = counter.load(std::memory_order_relaxed); current < value) {
        counter.store(value, std::memory_order_relaxed);
    }
}

[[nodiscard]] size_t HistogramBucket(u64 value) noexcept {
    if (value < HistogramExactValues) {
        return static_cast<size_t>(value);
    }
    const u32 exponent = std::bit_width(value) - 1;
    const u64 base = u64{1} << exponent;
    const u32 shift = exponent - 3;
    const u64 subdivision = (value - base) >> shift;
    return HistogramExactValues + (exponent - 3) * HistogramSubdivisions + subdivision;
}

[[nodiscard]] std::pair<u64, u64> HistogramBounds(size_t bucket) noexcept {
    if (bucket < HistogramExactValues) {
        return {bucket, bucket};
    }
    const size_t scaled = bucket - HistogramExactValues;
    const u32 exponent = 3 + static_cast<u32>(scaled / HistogramSubdivisions);
    const u64 subdivision = scaled % HistogramSubdivisions;
    const u64 width = u64{1} << (exponent - 3);
    const u64 lower = (u64{1} << exponent) + subdivision * width;
    return {lower, lower + width - 1};
}

void ObserveHistogramSingleWriter(ThreadRing& ring, Counter counter, u64 value) noexcept {
    const u8 histogram = HistogramIndices[static_cast<size_t>(counter)];
    if (histogram != NoHistogram) {
        AddSingleWriter(ring.histograms[histogram][HistogramBucket(value)], 1);
    }
}

void AddDurationSingleWriter(ThreadRing& ring, Counter counter, u64 duration) noexcept {
    AddSingleWriter(ring.counters[static_cast<size_t>(counter)], duration);
    ObserveHistogramSingleWriter(ring, counter, duration);
}

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
[[nodiscard]] bool ShouldSample(u32& sequence, u32 period) noexcept {
    const u32 mixed = ++sequence * 0x9E3779B9U;
    const u64 threshold = (u64{1} << 32) / period;
    return mixed < threshold;
}

template <size_t Size>
void AddMaskBits(std::array<std::atomic<u64>, Size>& counters, u32 mask) noexcept {
    while (mask != 0) {
        const u32 bit = std::countr_zero(mask);
        if (bit < Size) {
            AddSingleWriter(counters[bit], 1);
        }
        mask &= mask - 1;
    }
}

[[nodiscard]] size_t StagingSizeBucket(u64 bytes) noexcept {
    return bytes == 0 ? 0 : std::bit_width(bytes);
}
#endif

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
[[nodiscard]] constexpr u32 PackPm4Identity(Pm4Engine engine, u32 queue_id, u32 opcode,
                                             u32 depth) noexcept {
    return static_cast<u32>(engine) | ((queue_id & 0xff) << 8) | ((opcode & 0xff) << 16) |
           ((depth & 0xff) << 24);
}

struct Pm4Identity {
    u32 engine;
    u32 queue_id;
    u32 opcode;
    u32 depth;
};

[[nodiscard]] constexpr Pm4Identity UnpackPm4Identity(u64 identity) noexcept {
    return {
        .engine = static_cast<u32>(identity & 0xff),
        .queue_id = static_cast<u32>((identity >> 8) & 0xff),
        .opcode = static_cast<u32>((identity >> 16) & 0xff),
        .depth = static_cast<u32>((identity >> 24) & 0xff),
    };
}

[[nodiscard]] constexpr u64 MixPm4Hash(u64 value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

[[nodiscard]] constexpr u64 NonZeroPm4Hash(u64 value) noexcept {
    const u64 hash = MixPm4Hash(value);
    return hash != 0 ? hash : 1;
}

[[nodiscard]] constexpr size_t Pm4RegisterSpace(u32 opcode) noexcept {
    switch (opcode) {
    case 0x68:
        return 0;
    case 0x69:
        return 1;
    case 0x76:
        return 2;
    case 0x79:
        return 3;
    default:
        return Pm4RegisterSpaceCount;
    }
}

[[nodiscard]] constexpr const char* GetPm4OpcodeName(u32 opcode) noexcept {
    switch (opcode) {
    case 0x10: return "IT_NOP";
    case 0x11: return "IT_SET_BASE";
    case 0x12: return "IT_CLEAR_STATE";
    case 0x13: return "IT_INDEX_BUFFER_SIZE";
    case 0x15: return "IT_DISPATCH_DIRECT";
    case 0x16: return "IT_DISPATCH_INDIRECT";
    case 0x1D: return "IT_ATOMIC_GDS";
    case 0x1E: return "IT_ATOMIC";
    case 0x1F: return "IT_OCCLUSION_QUERY";
    case 0x20: return "IT_SET_PREDICATION";
    case 0x21: return "IT_REG_RMW";
    case 0x22: return "IT_COND_EXEC";
    case 0x23: return "IT_PRED_EXEC";
    case 0x24: return "IT_DRAW_INDIRECT";
    case 0x25: return "IT_DRAW_INDEX_INDIRECT";
    case 0x26: return "IT_INDEX_BASE";
    case 0x27: return "IT_DRAW_INDEX_2";
    case 0x28: return "IT_CONTEXT_CONTROL";
    case 0x2A: return "IT_INDEX_TYPE";
    case 0x2C: return "IT_DRAW_INDIRECT_MULTI";
    case 0x2D: return "IT_DRAW_INDEX_AUTO";
    case 0x2F: return "IT_NUM_INSTANCES";
    case 0x30: return "IT_DRAW_INDEX_MULTI_AUTO";
    case 0x33: return "IT_INDIRECT_BUFFER_CONST";
    case 0x34: return "IT_STRMOUT_BUFFER_UPDATE";
    case 0x35: return "IT_DRAW_INDEX_OFFSET_2";
    case 0x37: return "IT_WRITE_DATA";
    case 0x38: return "IT_DRAW_INDEX_INDIRECT_MULTI";
    case 0x39: return "IT_MEM_SEMAPHORE";
    case 0x3C: return "IT_WAIT_REG_MEM";
    case 0x3F: return "IT_INDIRECT_BUFFER";
    case 0x40: return "IT_COPY_DATA";
    case 0x41: return "IT_CP_DMA";
    case 0x42: return "IT_PFP_SYNC_ME";
    case 0x43: return "IT_SURFACE_SYNC";
    case 0x45: return "IT_COND_WRITE";
    case 0x46: return "IT_EVENT_WRITE";
    case 0x47: return "IT_EVENT_WRITE_EOP";
    case 0x48: return "IT_EVENT_WRITE_EOS";
    case 0x49: return "IT_RELEASE_MEM";
    case 0x4A: return "IT_PREAMBLE_CNTL";
    case 0x50: return "IT_DMA_DATA";
    case 0x51: return "IT_CONTEXT_REG_RMW";
    case 0x58: return "IT_ACQUIRE_MEM";
    case 0x59: return "IT_REWIND";
    case 0x5F: return "IT_LOAD_SH_REG";
    case 0x60: return "IT_LOAD_CONFIG_REG";
    case 0x61: return "IT_LOAD_CONTEXT_REG";
    case 0x68: return "IT_SET_CONFIG_REG";
    case 0x69: return "IT_SET_CONTEXT_REG";
    case 0x73: return "IT_SET_CONTEXT_REG_INDIRECT";
    case 0x76: return "IT_SET_SH_REG";
    case 0x77: return "IT_SET_SH_REG_OFFSET";
    case 0x78: return "IT_SET_QUEUE_REG";
    case 0x79: return "IT_SET_UCONFIG_REG";
    case 0x80: return "IT_LOAD_CONST_RAM";
    case 0x81: return "IT_WRITE_CONST_RAM";
    case 0x83: return "IT_DUMP_CONST_RAM";
    case 0x84: return "IT_INCREMENT_CE_COUNTER";
    case 0x85: return "IT_INCREMENT_DE_COUNTER";
    case 0x86: return "IT_WAIT_ON_CE_COUNTER";
    case 0x88: return "IT_WAIT_ON_DE_COUNTER_DIFF";
    case 0x8E: return "IT_GET_LOD_STATS";
    case 0x9D: return "IT_DRAW_INDEX_INDIRECT_COUNT_MULTI";
    default: return "IT_UNKNOWN";
    }
}

[[nodiscard]] constexpr const char* GetPm4EngineName(size_t engine) noexcept {
    switch (engine) {
    case 0: return "ConstantEngine";
    case 1: return "GraphicsEngine";
    case 2: return "ComputeEngine";
    default: return "UnknownEngine";
    }
}

[[nodiscard]] constexpr const char* GetPm4RegisterSpaceName(size_t space) noexcept {
    switch (space) {
    case 0: return "Config";
    case 1: return "Context";
    case 2: return "SH";
    case 3: return "UConfig";
    default: return "Unknown";
    }
}

[[nodiscard]] std::string GetPm4RegisterName(size_t space, u32 offset) {
    if (space == 1) { // Context registers (base 0xA000 / offset)
        switch (offset + 0xA000) {
        case 0xA000: return "DB_RENDER_CONTROL";
        case 0xA008: return "DB_COUNT_CONTROL";
        case 0xA00C: return "DB_DEPTH_VIEW";
        case 0xA010: return "DB_Z_INFO";
        case 0xA014: return "DB_STENCIL_INFO";
        case 0xA018: return "DB_Z_READ_BASE";
        case 0xA01C: return "DB_STENCIL_READ_BASE";
        case 0xA020: return "DB_Z_WRITE_BASE";
        case 0xA024: return "DB_STENCIL_WRITE_BASE";
        case 0xA080: return "PA_SC_WINDOW_OFFSET";
        case 0xA084: return "PA_SC_CLIPRECT_0_TL";
        case 0xA088: return "PA_SC_CLIPRECT_0_BR";
        case 0xA08C: return "PA_SC_CLIPRECT_1_TL";
        case 0xA090: return "PA_SC_CLIPRECT_1_BR";
        case 0xA094: return "PA_SC_CLIPRECT_2_TL";
        case 0xA098: return "PA_SC_CLIPRECT_2_BR";
        case 0xA09C: return "PA_SC_CLIPRECT_3_TL";
        case 0xA0A0: return "PA_SC_CLIPRECT_3_BR";
        case 0xA0A8: return "PA_SC_AA_MASK";
        case 0xA0B0: return "PA_SC_SCREEN_EXTENT";
        case 0xA0B4: return "PA_SC_GENERIC_SCISSOR_TL";
        case 0xA0B8: return "PA_SC_GENERIC_SCISSOR_BR";
        case 0xA0D0: return "PA_SC_VPORT_SCISSOR_0_TL";
        case 0xA0D4: return "PA_SC_VPORT_SCISSOR_0_BR";
        case 0xA100: return "PA_CL_CLIP_CNTL";
        case 0xA191: return "VGT_PRIMITIVE_TYPE";
        case 0xA192: return "VGT_INDEX_TYPE";
        case 0xA193: return "VGT_STRMOUT_CONFIG";
        case 0xA197: return "VGT_NUM_INDICES";
        case 0xA1B5: return "VGT_PRIMITIVEID_EN";
        case 0xA204: return "SPI_SHADER_PGM_RSRC1_PS";
        case 0xA205: return "SPI_SHADER_PGM_RSRC2_PS";
        case 0xA206: return "SPI_SHADER_USER_DATA_PS_0";
        case 0xA207: return "SPI_SHADER_Z_FORMAT";
        case 0xA208: return "SPI_SHADER_COL_FORMAT";
        case 0xA209: return "SPI_BARYC_CNTL";
        case 0xA20A: return "SPI_PS_INPUT_ENA";
        case 0xA20B: return "SPI_PS_INPUT_ADDR";
        case 0xA20C: return "SPI_INTERP_CONTROL_0";
        case 0xA280: return "CB_BLEND_RED";
        case 0xA281: return "CB_BLEND_GREEN";
        case 0xA282: return "CB_BLEND_BLUE";
        case 0xA283: return "CB_BLEND_ALPHA";
        case 0xA286: return "CB_BLEND0_CONTROL";
        case 0xA292: return "CB_COLOR_CONTROL";
        case 0xA293: return "CB_TARGET_MASK";
        case 0xA318: return "CB_COLOR0_BASE";
        case 0xA319: return "CB_COLOR0_PITCH";
        case 0xA31A: return "CB_COLOR0_SLICE";
        case 0xA31B: return "CB_COLOR0_VIEW";
        case 0xA31C: return "CB_COLOR0_INFO";
        case 0xA31D: return "CB_COLOR0_ATTRIB";
        case 0xA31F: return "CB_COLOR0_CMASK";
        case 0xA327: return "CB_COLOR1_BASE";
        case 0xA32C: return "CB_COLOR1_INFO";
        case 0xA32E: return "CB_COLOR1_CMASK";
        case 0xA336: return "CB_COLOR2_BASE";
        case 0xA33B: return "CB_COLOR2_INFO";
        case 0xA33D: return "CB_COLOR2_CMASK";
        case 0xA345: return "CB_COLOR3_BASE";
        case 0xA34A: return "CB_COLOR3_INFO";
        case 0xA34C: return "CB_COLOR3_CMASK";
        case 0xA354: return "CB_COLOR4_BASE";
        case 0xA359: return "CB_COLOR4_INFO";
        case 0xA35B: return "CB_COLOR4_CMASK";
        case 0xA363: return "CB_COLOR5_BASE";
        case 0xA368: return "CB_COLOR5_INFO";
        case 0xA36A: return "CB_COLOR5_CMASK";
        case 0xA372: return "CB_COLOR6_BASE";
        case 0xA377: return "CB_COLOR6_INFO";
        case 0xA379: return "CB_COLOR6_CMASK";
        case 0xA381: return "CB_COLOR7_BASE";
        case 0xA386: return "CB_COLOR7_INFO";
        case 0xA388: return "CB_COLOR7_CMASK";
        default:
            return "CONTEXT_0x" + std::format("{:04X}", offset + 0xA000);
        }
    } else if (space == 0) { // Config registers (base 0x8000)
        switch (offset + 0x8000) {
        case 0x85F0: return "CP_COHER_CNTL";
        case 0x85F8: return "CP_COHER_SIZE";
        case 0x85FC: return "CP_COHER_BASE";
        default:
            return "CONFIG_0x" + std::format("{:04X}", offset + 0x8000);
        }
    } else if (space == 2) { // SH registers (base 0x2C00)
        switch (offset + 0x2C00) {
        case 0x2C06: return "SPI_SHADER_PGM_RSRC1_VS";
        case 0x2C07: return "SPI_SHADER_PGM_RSRC2_VS";
        case 0x2C46: return "SPI_SHADER_PGM_RSRC1_GS";
        case 0x2C47: return "SPI_SHADER_PGM_RSRC2_GS";
        case 0x2C86: return "SPI_SHADER_PGM_RSRC1_ES";
        case 0x2C87: return "SPI_SHADER_PGM_RSRC2_ES";
        case 0x2CC6: return "SPI_SHADER_PGM_RSRC1_HS";
        case 0x2CC7: return "SPI_SHADER_PGM_RSRC2_HS";
        case 0x2D06: return "SPI_SHADER_PGM_RSRC1_LS";
        case 0x2D07: return "SPI_SHADER_PGM_RSRC2_LS";
        case 0x2E00: return "COMPUTE_PGM_RSRC1";
        case 0x2E01: return "COMPUTE_PGM_RSRC2";
        case 0x2E12: return "COMPUTE_NUM_THREAD_X";
        case 0x2E13: return "COMPUTE_NUM_THREAD_Y";
        case 0x2E14: return "COMPUTE_NUM_THREAD_Z";
        default:
            return "SH_0x" + std::format("{:04X}", offset + 0x2C00);
        }
    } else if (space == 3) { // UConfig registers (base 0xC000)
        return "UCONFIG_0x" + std::format("{:04X}", offset + 0xC000);
    }
    return "REG_0x" + std::format("{:04X}", offset);
}
#endif

void WriteEvent(ThreadRing& ring, EventType type, u64 arg0, u64 arg1) noexcept {
    const u64 sequence = ring.next_sequence.load(std::memory_order_relaxed);
    ring.next_sequence.store(sequence + 1, std::memory_order_release);
    auto& slot = ring.events[sequence & RingMask];
    slot.timestamp_ns.store(Timestamp(), std::memory_order_relaxed);
    slot.arg0.store(arg0, std::memory_order_relaxed);
    slot.arg1.store(arg1, std::memory_order_relaxed);
    slot.metadata.store(static_cast<u64>(type) | (static_cast<u64>(ring.id) << 16),
                        std::memory_order_relaxed);
    slot.committed_sequence.store(sequence + 1, std::memory_order_release);
}

[[nodiscard]] bool IsMaxCounter(Counter counter) noexcept {
    return counter == Counter::IbDepthMax || counter == Counter::SubmitQueueDepthMax ||
           counter == Counter::ShaderModuleQueueDepthMax;
}

[[nodiscard]] std::string CsvSafe(std::string value) {
    std::ranges::replace(value, ',', '_');
    std::ranges::replace(value, '\n', '_');
    std::ranges::replace(value, '\r', '_');
    return value;
}

} // namespace

TraceCaptureProfile GetCaptureProfileEnabled() noexcept {
    static const TraceCaptureProfile profile = []() {
        if (const char* env = std::getenv("SHADPS4_CAPTURE_PROFILE")) {
            if (std::strcmp(env, "sync_perf") == 0 || std::strcmp(env, "0") == 0) {
                return TraceCaptureProfile::SyncPerf;
            }
            if (std::strcmp(env, "sync_semantic") == 0 || std::strcmp(env, "1") == 0) {
                return TraceCaptureProfile::SyncSemantic;
            }
            if (std::strcmp(env, "sync_fastpath_validation") == 0 || std::strcmp(env, "sync_fastpath") == 0 || std::strcmp(env, "2") == 0) {
                return TraceCaptureProfile::SyncFastpathValidation;
            }
        }
        if (const char* env = std::getenv("SHADPS4_TELEMETRY_PROFILE")) {
            if (std::strcmp(env, "sync_perf") == 0 || std::strcmp(env, "0") == 0) {
                return TraceCaptureProfile::SyncPerf;
            }
            if (std::strcmp(env, "sync_semantic") == 0 || std::strcmp(env, "1") == 0) {
                return TraceCaptureProfile::SyncSemantic;
            }
            if (std::strcmp(env, "sync_fastpath_validation") == 0 || std::strcmp(env, "sync_fastpath") == 0 || std::strcmp(env, "2") == 0) {
                return TraceCaptureProfile::SyncFastpathValidation;
            }
        }
        return TraceCaptureProfile::SyncFastpathValidation;
    }();
    return profile;
}

void AddEnabled(Counter counter, u64 value) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(counter)], value);
    }
}

void ObserveMaxEnabled(Counter counter, u64 value) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        ObserveSingleWriterMax(ring->counters[static_cast<size_t>(counter)], value);
        ObserveHistogramSingleWriter(*ring, counter, value);
    }
}

void RecordEnabled(EventType type, u64 arg0, u64 arg1) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        WriteEvent(*ring, type, arg0, arg1);
    }
}

void RecordDurationEnabled(Counter counter, EventType type, u64 start_ns, u64 arg0) noexcept {
    if (!Enabled() || g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        const u64 duration = Timestamp() - start_ns;
        AddDurationSingleWriter(*ring, counter, duration);
        if (type != EventType::None) {
            WriteEvent(*ring, type, arg0, duration);
        }
    }
}

void RecordDurationValueEnabled(Counter counter, EventType type, u64 duration, u64 arg0) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddDurationSingleWriter(*ring, counter, duration);
        if (type != EventType::None) {
            WriteEvent(*ring, type, arg0, duration);
        }
    }
}

void RecordTimerSampleEnabled(TimerSite site, u64 start_ns, u32 stage) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    RecordTimerSampleDurationEnabled(site, Timestamp() - start_ns, stage);
#else
    static_cast<void>(site);
    static_cast<void>(start_ns);
    static_cast<void>(stage);
#endif
}

void RecordTimerSampleDurationEnabled(TimerSite site, u64 duration_ns, u32 stage) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t site_index = static_cast<size_t>(site);
    if (site_index >= TimerSiteCount) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->timer_ns[site_index], duration_ns);
        AddSingleWriter(ring->timer_samples[site_index], 1);
        AddSingleWriter(ring->timer_histograms[site_index][HistogramBucket(duration_ns)], 1);
        if (stage < TrackedStageCount) {
            AddSingleWriter(ring->stage_timer_ns[stage][site_index], duration_ns);
            AddSingleWriter(ring->stage_timer_samples[stage][site_index], 1);
        }
    }
#else
    static_cast<void>(site);
    static_cast<void>(duration_ns);
    static_cast<void>(stage);
#endif
}

void RecordStageUncacheableEnabled(u32 stage, u32 reason_mask) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(
            ring->counters[static_cast<size_t>(Counter::StageCacheUncacheable)], 1);
        static thread_local u32 sequence{};
        if (stage < TrackedStageCount && ShouldSample(sequence, StageReasonSamplePeriod)) {
            AddSingleWriter(ring->stage_reasons[stage][reason_mask & (StageReasonCount - 1)], 1);
            AddMaskBits(ring->stage_reason_bits[stage], reason_mask);
        }
    }
#else
    static_cast<void>(stage);
    static_cast<void>(reason_mask);
#endif
}

void RecordStageSlowResultEnabled(bool hit, bool current, u64 comparisons) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(
            ring->counters[static_cast<size_t>(Counter::StageSlowSearchComparisons)],
            comparisons);
        if (hit) {
            AddSingleWriter(ring->counters[static_cast<size_t>(Counter::StagePermutationHits)], 1);
            AddSingleWriter(
                ring->counters[static_cast<size_t>(
                    current ? Counter::StageSlowCurrentHits : Counter::StageSlowOtherHits)],
                1);
        }
    }
#else
    static_cast<void>(hit);
    static_cast<void>(current);
    static_cast<void>(comparisons);
#endif
}

void RecordDynamicStateDecisionEnabled(u32 reason_mask) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(
                            reason_mask == 0 ? Counter::DynamicStateHits
                                             : Counter::DynamicStateMisses)],
                        1);
        if (reason_mask != 0) {
            static thread_local u32 sequence{};
            if (ShouldSample(sequence, DynamicReasonSamplePeriod)) {
                AddSingleWriter(ring->dynamic_reasons[reason_mask & (DynamicReasonCount - 1)], 1);
                AddMaskBits(ring->dynamic_reason_bits, reason_mask);
            }
        }
    }
#else
    static_cast<void>(reason_mask);
#endif
}

void RecordDynamicCommitEnabled(u32 pending_groups, u32 emitted_groups) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        if (pending_groups == 0) {
            AddSingleWriter(
                ring->counters[static_cast<size_t>(Counter::DynamicStateEmptyCommits)], 1);
            return;
        }
        static thread_local u32 sequence{};
        if (ShouldSample(sequence, DynamicReasonSamplePeriod)) {
            AddSingleWriter(ring->dynamic_pending_groups[pending_groups &
                                                        (DynamicGroupCount - 1)],
                            1);
            AddSingleWriter(ring->dynamic_emitted_groups[emitted_groups &
                                                        (DynamicGroupCount - 1)],
                            1);
            AddMaskBits(ring->dynamic_pending_group_bits, pending_groups);
            AddMaskBits(ring->dynamic_emitted_group_bits, emitted_groups);
        }
    }
#else
    static_cast<void>(pending_groups);
    static_cast<void>(emitted_groups);
#endif
}

void RecordDescriptorDecisionEnabled(u32 reason_mask) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(
                            reason_mask == 0 ? Counter::DescriptorHits
                                             : Counter::DescriptorMisses)],
                        1);
        if (reason_mask != 0) {
            static thread_local u32 sequence{};
            if (ShouldSample(sequence, DescriptorReasonSamplePeriod)) {
                AddSingleWriter(
                    ring->descriptor_reasons[reason_mask & (DescriptorReasonCount - 1)], 1);
                AddMaskBits(ring->descriptor_reason_bits, reason_mask);
            }
        }
    }
#else
    static_cast<void>(reason_mask);
#endif
}

void RecordImageFindPathEnabled(ImageFindPath path) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t path_index = static_cast<size_t>(path);
    if (path_index >= ImageFindPathCount) {
        return;
    }
    static thread_local u32 sequence{};
    if (ShouldSample(sequence, ImageFindPathSamplePeriod)) {
        if (auto* ring = GetThreadRing()) {
            AddSingleWriter(ring->image_find_paths[path_index], 1);
        }
    }
#else
    static_cast<void>(path);
#endif
}

void RecordStagingAllocationEnabled(StagingSite site, u64 bytes) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t site_index = static_cast<size_t>(site);
    if (site_index >= StagingSiteCount) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::StagingBytes)], bytes);
        static thread_local u32 sequence{};
        if (ShouldSample(sequence, StagingDetailSamplePeriod)) {
            auto& detail = ring->staging[site_index];
            AddSingleWriter(detail.allocations, 1);
            AddSingleWriter(detail.bytes, bytes);
            AddSingleWriter(detail.sizes[StagingSizeBucket(bytes)], 1);
        }
    }
#else
    static_cast<void>(site);
    static_cast<void>(bytes);
#endif
}

void RecordStagingSourceEnabled(StagingSite site, StagingSource source, u64 bytes) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t site_index = static_cast<size_t>(site);
    const size_t source_index = static_cast<size_t>(source);
    if (site_index >= StagingSiteCount || source_index >= StagingSourceCount) {
        return;
    }
    static thread_local u32 sequence{};
    if (ShouldSample(sequence, StagingDetailSamplePeriod)) {
        if (auto* ring = GetThreadRing()) {
            auto& detail = ring->staging[site_index];
            AddSingleWriter(detail.source_records[source_index], 1);
            AddSingleWriter(detail.source_bytes[source_index], bytes);
        }
    }
#else
    static_cast<void>(site);
    static_cast<void>(source);
    static_cast<void>(bytes);
#endif
}

bool ShouldSampleStagingBatchEnabled() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return false;
    }
    static thread_local u32 sequence{};
    return ShouldSample(sequence, StagingDetailSamplePeriod);
#else
    return false;
#endif
}

void RecordStagingBatchEnabled(const StagingBatchSample& sample, bool sampled) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::StagingBytes)],
                        sample.allocated_bytes);
        if (!sampled) {
            return;
        }

        const auto add_counter = [&](Counter counter, u64 value) {
            AddSingleWriter(ring->counters[static_cast<size_t>(counter)], value);
        };
        add_counter(Counter::StagingBatchSamples, 1);
        add_counter(Counter::StagingBatchRequests, sample.requests);
        add_counter(Counter::StagingBatchCanonicalCopies, sample.canonical_copies);
        add_counter(Counter::StagingBatchGuestCopies, sample.guest_copies);
        add_counter(Counter::StagingBatchExactReuses, sample.exact_reuses);
        add_counter(Counter::StagingBatchSubrangeReuses, sample.subrange_reuses);
        add_counter(Counter::StagingBatchRequestedBytes, sample.requested_bytes);
        add_counter(Counter::StagingBatchCanonicalBytes, sample.canonical_bytes);
        add_counter(Counter::StagingBatchAllocatedBytes, sample.allocated_bytes);
        add_counter(Counter::StagingReuseCandidates, sample.reuse_candidates);
        add_counter(Counter::StagingReuseHits, sample.reuse_hits);
        add_counter(Counter::StagingReuseMismatches, sample.reuse_mismatches);
        add_counter(Counter::StagingReuseCooldownSkips, sample.reuse_cooldown_skips);
        add_counter(Counter::StagingReuseInvalidations, sample.reuse_invalidations);
        add_counter(Counter::StagingReuseWarmups, sample.reuse_warmups);
        add_counter(Counter::StagingReuseCandidateBytes, sample.reuse_candidate_bytes);
        add_counter(Counter::StagingReuseAvoidedBytes, sample.reuse_avoided_bytes);
        add_counter(Counter::StagingReuseWarmupBytes, sample.reuse_warmup_bytes);

        auto& detail = ring->staging[static_cast<size_t>(StagingSite::StreamBatch)];
        AddSingleWriter(detail.allocations, 1);
        AddSingleWriter(detail.bytes, sample.allocated_bytes);
        AddSingleWriter(detail.sizes[StagingSizeBucket(sample.allocated_bytes)], 1);
        const std::array source_records{sample.guest_copies, sample.host_copies,
                                        sample.zero_copies};
        const std::array source_bytes{sample.guest_bytes, sample.host_bytes,
                                      sample.zero_bytes};
        for (size_t source = 0; source < StagingSourceCount; ++source) {
            AddSingleWriter(detail.source_records[source], source_records[source]);
            AddSingleWriter(detail.source_bytes[source], source_bytes[source]);
        }
    }
#else
    static_cast<void>(sample);
    static_cast<void>(sampled);
#endif
}

void RecordStagingSparseCopyEnabled(const StagingSparseCopySample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        const auto add_counter = [&](Counter counter, u64 value) {
            AddSingleWriter(ring->counters[static_cast<size_t>(counter)], value);
        };
        add_counter(Counter::StagingSparseCopySamples, 1);
        add_counter(Counter::StagingSparseCopyRequests, sample.requests);
        add_counter(Counter::StagingSparsePlanHits, sample.plan_hits);
        add_counter(Counter::StagingSparsePlanMisses, sample.plan_misses);
        add_counter(Counter::StagingSparsePlansBuilt, sample.plans_built);
        add_counter(Counter::StagingSparseCopyReferenceFallbacks,
                    sample.copy_reference_fallbacks);
        add_counter(Counter::StagingSparseMappedRuns, sample.mapped_runs);
        add_counter(Counter::StagingSparseZeroRuns, sample.zero_runs);
        add_counter(Counter::StagingSparseRequestedBytes, sample.requested_bytes);
        add_counter(Counter::StagingSparseCopiedBytes, sample.copied_bytes);
        add_counter(Counter::StagingSparseReferenceBytes, sample.reference_bytes);
        add_counter(Counter::StagingSparseMappedBytes, sample.mapped_bytes);
        add_counter(Counter::StagingSparseZeroBytes, sample.zero_bytes);
        add_counter(Counter::StagingSparsePlanMissEmpty, sample.plan_miss_empty);
        add_counter(Counter::StagingSparsePlanMissGeneration, sample.plan_miss_generation);
        add_counter(Counter::StagingSparsePlanMissConflict, sample.plan_miss_conflict);
        add_counter(Counter::StagingSparsePlanMissUncacheable, sample.plan_miss_uncacheable);
        add_counter(Counter::StagingSparseNonTemporalRuns, sample.non_temporal_runs);
        add_counter(Counter::StagingSparseNonTemporalBytes, sample.non_temporal_bytes);
        add_counter(Counter::StagingSparseNonTemporalFences, sample.non_temporal_fences);
        add_counter(Counter::StagingSparseMergeablePairs, sample.mergeable_pairs);
        add_counter(Counter::StagingSparseMergeableBytes, sample.mergeable_bytes);
        add_counter(Counter::StagingSparseDenseHotHits, sample.dense_hot_hits);
        add_counter(Counter::StagingSparseDenseCacheHits, sample.dense_cache_hits);
        add_counter(Counter::StagingSparseDenseRefills, sample.dense_refills);
        add_counter(Counter::StagingSparseDenseFallbackRequests,
                    sample.dense_fallback_requests);
        add_counter(Counter::StagingSparseDenseHotBytes, sample.dense_hot_bytes);
        add_counter(Counter::StagingSparseDenseCacheBytes, sample.dense_cache_bytes);
        add_counter(Counter::StagingSparseDenseFallbackBytes, sample.dense_fallback_bytes);
    }
#else
    static_cast<void>(sample);
#endif
}

void RecordStagingBackendEnabled(StagingBackend backend, u64 bytes) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        Counter batches{};
        Counter byte_counter{};
        switch (backend) {
        case StagingBackend::CurrentStream:
            batches = Counter::StagingCurrentStreamBatches;
            byte_counter = Counter::StagingCurrentStreamBytes;
            break;
        case StagingBackend::HostDirect:
            batches = Counter::StagingHostDirectBatches;
            byte_counter = Counter::StagingHostDirectBytes;
            break;
        case StagingBackend::GpuPromoted:
            batches = Counter::StagingGpuPromotedBatches;
            byte_counter = Counter::StagingGpuPromotedBytes;
            break;
        case StagingBackend::Count:
            return;
        }
        AddSingleWriter(ring->counters[static_cast<size_t>(batches)], 1);
        AddSingleWriter(ring->counters[static_cast<size_t>(byte_counter)], bytes);
    }
#else
    static_cast<void>(backend);
    static_cast<void>(bytes);
#endif
}

void RecordStagingMemoryTypeEnabled(StagingMemoryKind kind, u32 memory_type, u32 memory_heap,
                                    u32 property_flags) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    const size_t index = static_cast<size_t>(kind);
    if (index >= StagingMemoryKindCount) {
        return;
    }
    auto& detail = g_staging_memory_types[index];
    detail.memory_type.store(memory_type, std::memory_order_relaxed);
    detail.memory_heap.store(memory_heap, std::memory_order_relaxed);
    detail.property_flags.store(property_flags, std::memory_order_relaxed);
    detail.valid.store(true, std::memory_order_release);
#else
    static_cast<void>(kind);
    static_cast<void>(memory_type);
    static_cast<void>(memory_heap);
    static_cast<void>(property_flags);
#endif
}

bool ShouldSampleStagingSparsePhaseEnabled() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return false;
    }
    constexpr u32 period = TimerSamplePeriod(TimerSite::StagingSparsePlanLookup);
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparseSharedTotal));
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparseLock));
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparsePlanBuild));
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparsePayloadCopy));
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparseFinish));
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparseDenseLookup));
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparseSpanRefill));
    static_assert(period == TimerSamplePeriod(TimerSite::StagingSparseColdPath));
    static thread_local u32 sequence{};
    return ShouldSample(sequence, period);
#else
    return false;
#endif
}

bool ShouldSampleDescriptorCrossPipelineEnabled() noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return false;
    }
    static thread_local u32 sequence{};
    return ShouldSample(sequence, DescriptorCrossPipelineSamplePeriod);
#else
    return false;
#endif
}

void RecordDescriptorCrossPipelineEnabled(bool exact_state, bool compatible_layout) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        const auto add = [&](Counter counter) {
            AddSingleWriter(ring->counters[static_cast<size_t>(counter)], 1);
        };
        add(Counter::DescriptorCrossPipelineSamples);
        if (exact_state) {
            add(Counter::DescriptorCrossPipelineExactStateHits);
        }
        if (compatible_layout) {
            add(Counter::DescriptorCrossPipelineCompatibleLayoutHits);
        }
        if (exact_state && compatible_layout) {
            add(Counter::DescriptorCrossPipelineReusableHits);
        }
    }
#else
    static_cast<void>(exact_state);
    static_cast<void>(compatible_layout);
#endif
}

void RecordSubmitTimingEnabled(SubmitReason reason, u64 mutex_wait_ns, u64 prepare_ns,
                               u64 driver_ns, u64 post_ns, u64 mutex_hold_ns) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t reason_index = static_cast<size_t>(reason);
    if (reason_index >= SubmitReasonCount) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddDurationSingleWriter(*ring, Counter::SubmitMutexWaitNs, mutex_wait_ns);
        AddDurationSingleWriter(*ring, Counter::SubmitMutexHoldNs, mutex_hold_ns);
        AddDurationSingleWriter(*ring, Counter::SubmitPrepareNs, prepare_ns);
        AddDurationSingleWriter(*ring, Counter::SubmitPostNs, post_ns);
        auto& detail = ring->submits[reason_index];
        AddSingleWriter(detail.calls, 1);
        AddSingleWriter(detail.mutex_wait_ns, mutex_wait_ns);
        AddSingleWriter(detail.prepare_ns, prepare_ns);
        AddSingleWriter(detail.driver_ns, driver_ns);
        AddSingleWriter(detail.post_ns, post_ns);
        AddSingleWriter(detail.mutex_hold_ns, mutex_hold_ns);
    }
#else
    static_cast<void>(reason);
    static_cast<void>(mutex_wait_ns);
    static_cast<void>(prepare_ns);
    static_cast<void>(driver_ns);
    static_cast<void>(post_ns);
    static_cast<void>(mutex_hold_ns);
#endif
}

void RecordPresentTimingEnabled(u64 mutex_wait_ns, u64 driver_ns, u64 mutex_hold_ns) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddDurationSingleWriter(*ring, Counter::PresentMutexWaitNs, mutex_wait_ns);
        AddDurationSingleWriter(*ring, Counter::PresentMutexHoldNs, mutex_hold_ns);
        auto& detail = ring->submits[static_cast<size_t>(SubmitReason::QueuePresent)];
        AddSingleWriter(detail.calls, 1);
        AddSingleWriter(detail.mutex_wait_ns, mutex_wait_ns);
        AddSingleWriter(detail.driver_ns, driver_ns);
        AddSingleWriter(detail.mutex_hold_ns, mutex_hold_ns);
    }
#else
    static_cast<void>(mutex_wait_ns);
    static_cast<void>(driver_ns);
    static_cast<void>(mutex_hold_ns);
#endif
}

void RecordWritebackDrainEnabled(WritebackTrigger trigger, u32 candidates,
                                 u32 scheduled) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t trigger_index = static_cast<size_t>(trigger);
    if (trigger_index >= WritebackTriggerCount) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::WritebackDrainCalls)], 1);
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::WritebackCandidates)],
                        candidates);
        if (scheduled == 0) {
            AddSingleWriter(ring->counters[static_cast<size_t>(Counter::WritebackDrainEmpty)], 1);
        }
        auto& detail = ring->writeback_drains[trigger_index];
        AddSingleWriter(detail[0], 1);
        AddSingleWriter(detail[1], candidates);
        AddSingleWriter(detail[2], scheduled);
    }
#else
    static_cast<void>(trigger);
    static_cast<void>(candidates);
    static_cast<void>(scheduled);
#endif
}

void RecordWritebackImageEnabled(const WritebackImageSample& sample) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    EnsureRelationalStorageInitialized();
    if (!g_writeback_records) {
        return;
    }
    if ((sample.flags & WritebackFlagSameEpoch) != 0) {
        if (auto* ring = GetThreadRing()) {
            AddSingleWriter(ring->counters[static_cast<size_t>(Counter::WritebackSameEpoch)], 1);
        }
    }
    const u64 sequence = g_writeback_sequence.fetch_add(1, std::memory_order_relaxed);
    auto& record = (*g_writeback_records)[sequence & (WritebackRecordCapacity - 1)];
    record.timestamp_ns = Timestamp();
    record.sample = sample;
    record.committed_sequence.store(sequence + 1, std::memory_order_release);
#else
    static_cast<void>(sample);
#endif
}

void RecordEventQueryEnabled(u32 counter_pairs, bool had_active_watches,
                             u32 matched_pages, u32 callbacks) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::EventQueryCalls)], 1);
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::EventQueryCounterPairs)],
                        counter_pairs);
        AddSingleWriter(ring->counters[static_cast<size_t>(
                            had_active_watches ? Counter::EventQueryActiveWatchCalls
                                               : Counter::EventQueryNoWatchCalls)],
                        1);
        if (matched_pages != 0 || callbacks != 0) {
            AddSingleWriter(
                ring->counters[static_cast<size_t>(Counter::EventQueryMatchedWatchCalls)], 1);
        }
    }
#else
    static_cast<void>(counter_pairs);
    static_cast<void>(had_active_watches);
    static_cast<void>(matched_pages);
    static_cast<void>(callbacks);
#endif
}

void RecordFrameSampleEnabled(u32 frame_id, u64 present_start_ns) noexcept {
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    EnsureRelationalStorageInitialized();
    if (!g_frame_records) {
        return;
    }

    std::array<u64, FrameCounters.size()> counters{};
    std::array<u64, TimerSiteCount> timer_ns{};
    std::array<u64, TimerSiteCount> timer_samples{};
    FrameDetailSnapshot details{};
    {
        std::scoped_lock lock{g_rings_mutex};
        for (const auto& ring : g_rings) {
            for (size_t index = 0; index < FrameCounters.size(); ++index) {
                counters[index] += ring->counters[static_cast<size_t>(FrameCounters[index])]
                                       .load(std::memory_order_relaxed);
            }
            for (size_t site = 0; site < TimerSiteCount; ++site) {
                timer_ns[site] += ring->timer_ns[site].load(std::memory_order_relaxed);
                timer_samples[site] +=
                    ring->timer_samples[site].load(std::memory_order_relaxed);
            }
            for (size_t stage = 0; stage < TrackedStageCount; ++stage) {
                for (size_t site = 0; site < StageFrameTimerCount; ++site) {
                    details.stage_timer_ns[stage][site] +=
                        ring->stage_timer_ns[stage][site].load(std::memory_order_relaxed);
                    details.stage_timer_samples[stage][site] +=
                        ring->stage_timer_samples[stage][site].load(std::memory_order_relaxed);
                }
                for (size_t bit = 0; bit < StageReasonBitCount; ++bit) {
                    details.stage_reason_bits[stage][bit] +=
                        ring->stage_reason_bits[stage][bit].load(std::memory_order_relaxed);
                }
            }
            for (size_t bit = 0; bit < DynamicReasonBitCount; ++bit) {
                details.dynamic_reason_bits[bit] +=
                    ring->dynamic_reason_bits[bit].load(std::memory_order_relaxed);
            }
            for (size_t bit = 0; bit < DynamicGroupBitCount; ++bit) {
                details.dynamic_pending_group_bits[bit] +=
                    ring->dynamic_pending_group_bits[bit].load(std::memory_order_relaxed);
                details.dynamic_emitted_group_bits[bit] +=
                    ring->dynamic_emitted_group_bits[bit].load(std::memory_order_relaxed);
            }
            for (size_t bit = 0; bit < DescriptorReasonBitCount; ++bit) {
                details.descriptor_reason_bits[bit] +=
                    ring->descriptor_reason_bits[bit].load(std::memory_order_relaxed);
            }
            for (size_t path = 0; path < ImageFindPathCount; ++path) {
                details.image_find_paths[path] +=
                    ring->image_find_paths[path].load(std::memory_order_relaxed);
            }
            for (size_t site = 0; site < StagingSiteCount; ++site) {
                const auto& staging = ring->staging[site];
                details.staging_allocations[site] +=
                    staging.allocations.load(std::memory_order_relaxed);
                details.staging_bytes[site] += staging.bytes.load(std::memory_order_relaxed);
                for (size_t source = 0; source < StagingSourceCount; ++source) {
                    details.staging_source_records[site][source] +=
                        staging.source_records[source].load(std::memory_order_relaxed);
                    details.staging_source_bytes[site][source] +=
                        staging.source_bytes[source].load(std::memory_order_relaxed);
                }
            }
            for (size_t reason = 0; reason < SubmitReasonCount; ++reason) {
                const auto& source = ring->submits[reason];
                auto& destination = details.submits[reason];
                destination.calls += source.calls.load(std::memory_order_relaxed);
                destination.mutex_wait_ns +=
                    source.mutex_wait_ns.load(std::memory_order_relaxed);
                destination.prepare_ns += source.prepare_ns.load(std::memory_order_relaxed);
                destination.driver_ns += source.driver_ns.load(std::memory_order_relaxed);
                destination.post_ns += source.post_ns.load(std::memory_order_relaxed);
                destination.mutex_hold_ns +=
                    source.mutex_hold_ns.load(std::memory_order_relaxed);
            }
        }
    }

    const u64 now = Timestamp();
    std::scoped_lock lock{g_frame_mutex};
    const u64 sequence = g_frame_sequence++;
    auto& record = (*g_frame_records)[sequence & (FrameRecordCapacity - 1)];
    record = {};
    record.sequence = sequence;
    record.timestamp_ns = now;
    record.interval_ns = g_previous_frame_timestamp == 0 ? 0 : now - g_previous_frame_timestamp;
    record.present_duration_ns = present_start_ns == 0 ? 0 : now - present_start_ns;
    record.frame_id = frame_id;
    for (size_t index = 0; index < counters.size(); ++index) {
        record.counters[index] = counters[index] - g_previous_frame_counters[index];
    }
    for (size_t site = 0; site < TimerSiteCount; ++site) {
        record.timer_ns[site] = timer_ns[site] - g_previous_frame_timer_ns[site];
        record.timer_samples[site] =
            timer_samples[site] - g_previous_frame_timer_samples[site];
    }
    for (size_t stage = 0; stage < TrackedStageCount; ++stage) {
        for (size_t site = 0; site < StageFrameTimerCount; ++site) {
            record.details.stage_timer_ns[stage][site] =
                details.stage_timer_ns[stage][site] -
                g_previous_frame_details.stage_timer_ns[stage][site];
            record.details.stage_timer_samples[stage][site] =
                details.stage_timer_samples[stage][site] -
                g_previous_frame_details.stage_timer_samples[stage][site];
        }
        for (size_t bit = 0; bit < StageReasonBitCount; ++bit) {
            record.details.stage_reason_bits[stage][bit] =
                details.stage_reason_bits[stage][bit] -
                g_previous_frame_details.stage_reason_bits[stage][bit];
        }
    }
    for (size_t bit = 0; bit < DynamicReasonBitCount; ++bit) {
        record.details.dynamic_reason_bits[bit] =
            details.dynamic_reason_bits[bit] -
            g_previous_frame_details.dynamic_reason_bits[bit];
    }
    for (size_t bit = 0; bit < DynamicGroupBitCount; ++bit) {
        record.details.dynamic_pending_group_bits[bit] =
            details.dynamic_pending_group_bits[bit] -
            g_previous_frame_details.dynamic_pending_group_bits[bit];
        record.details.dynamic_emitted_group_bits[bit] =
            details.dynamic_emitted_group_bits[bit] -
            g_previous_frame_details.dynamic_emitted_group_bits[bit];
    }
    for (size_t bit = 0; bit < DescriptorReasonBitCount; ++bit) {
        record.details.descriptor_reason_bits[bit] =
            details.descriptor_reason_bits[bit] -
            g_previous_frame_details.descriptor_reason_bits[bit];
    }
    for (size_t path = 0; path < ImageFindPathCount; ++path) {
        record.details.image_find_paths[path] =
            details.image_find_paths[path] - g_previous_frame_details.image_find_paths[path];
    }
    for (size_t site = 0; site < StagingSiteCount; ++site) {
        record.details.staging_allocations[site] =
            details.staging_allocations[site] -
            g_previous_frame_details.staging_allocations[site];
        record.details.staging_bytes[site] =
            details.staging_bytes[site] - g_previous_frame_details.staging_bytes[site];
        for (size_t source = 0; source < StagingSourceCount; ++source) {
            record.details.staging_source_records[site][source] =
                details.staging_source_records[site][source] -
                g_previous_frame_details.staging_source_records[site][source];
            record.details.staging_source_bytes[site][source] =
                details.staging_source_bytes[site][source] -
                g_previous_frame_details.staging_source_bytes[site][source];
        }
    }
    for (size_t reason = 0; reason < SubmitReasonCount; ++reason) {
        const auto& current = details.submits[reason];
        const auto& previous = g_previous_frame_details.submits[reason];
        record.details.submits[reason] = {
            .calls = current.calls - previous.calls,
            .mutex_wait_ns = current.mutex_wait_ns - previous.mutex_wait_ns,
            .prepare_ns = current.prepare_ns - previous.prepare_ns,
            .driver_ns = current.driver_ns - previous.driver_ns,
            .post_ns = current.post_ns - previous.post_ns,
            .mutex_hold_ns = current.mutex_hold_ns - previous.mutex_hold_ns,
        };
    }
    g_previous_frame_timestamp = now;
    g_previous_frame_counters = counters;
    g_previous_frame_timer_ns = timer_ns;
    g_previous_frame_timer_samples = timer_samples;
    g_previous_frame_details = details;
#else
    static_cast<void>(frame_id);
    static_cast<void>(present_start_ns);
#endif
}

void CountPm4PacketEnabled(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                           uintptr_t address, u32 words, u32 header) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        AddSingleWriter(ring->counters[static_cast<size_t>(Counter::Pm4Packets)], 1);
        AddSingleWriter(ring->opcodes[opcode & 0xff], 1);
        ObserveSingleWriterMax(ring->counters[static_cast<size_t>(Counter::IbDepthMax)], depth + 1);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        const size_t engine_index = static_cast<size_t>(engine);
        auto& detail = ring->pm4.opcodes[engine_index][opcode & 0xff];
        AddSingleWriter(detail.packets, 1);
        AddSingleWriter(detail.words, words);
        AddSingleWriter(detail.predicated, header & 1);
        AddSingleWriter(detail.shader_compute, (header >> 1) & 1);
        ObserveSingleWriterMax(detail.max_depth, depth);
        AddSingleWriter(detail.word_counts[std::min<size_t>(words, Pm4WordOverflowBucket)], 1);
#else
        static_cast<void>(engine);
        static_cast<void>(queue_id);
        static_cast<void>(words);
        static_cast<void>(header);
#endif
        static_cast<void>(address);
    }
}

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
void RecordPm4ControlEnabled(Pm4Engine engine, u32 queue_id, u32 opcode, u32 depth,
                             u32 control0, u32 control1, u32 tag) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    auto* ring = GetThreadRing();
    if (ring == nullptr) {
        return;
    }

    const u64 identity = PackPm4Identity(engine, queue_id, opcode, depth) |
                         (static_cast<u64>(tag) << 32);
    const u64 hash = NonZeroPm4Hash(
        NonZeroPm4Hash(static_cast<u64>(identity) | (static_cast<u64>(control0) << 32)) ^
        MixPm4Hash(control1));
    size_t slot = hash & (Pm4ControlCapacity - 1);
    for (size_t probe = 0; probe < Pm4HashProbeLimit; ++probe) {
        auto& entry = ring->pm4.controls[slot];
        const u64 entry_hash = entry.hash.load(std::memory_order_acquire);
        if (entry_hash == hash && entry.identity == identity && entry.control0 == control0 &&
            entry.control1 == control1) {
            AddSingleWriter(entry.packets, 1);
            return;
        }
        if (entry_hash == 0) {
            entry.identity = identity;
            entry.control0 = control0;
            entry.control1 = control1;
            entry.hash.store(hash, std::memory_order_release);
            AddSingleWriter(entry.packets, 1);
            return;
        }
        slot = (slot + 1) & (Pm4ControlCapacity - 1);
    }
    AddSingleWriter(ring->pm4.control_overflow, 1);
}

void RecordPm4RegisterEnabled(Pm4Engine engine, u32 opcode, u32 register_offset, u32 words,
                              bool changed) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    auto* ring = GetThreadRing();
    if (ring == nullptr) {
        return;
    }
    const size_t engine_index = static_cast<size_t>(engine);
    const size_t space = Pm4RegisterSpace(opcode);
    if (engine_index >= Pm4EngineCount || space >= Pm4RegisterSpaceCount ||
        register_offset >= Pm4RegisterCount) {
        AddSingleWriter(ring->pm4.register_overflow, 1);
        return;
    }
    auto& detail = ring->pm4.registers[engine_index][space][register_offset];
    AddSingleWriter(detail.packets, 1);
    AddSingleWriter(detail.words, words);
    AddSingleWriter(detail.changed, changed);
}

void RecordPm4WaitEnabled(Pm4Engine engine, u32 queue_id, u32 depth, u32 control,
                          u64 location, u32 reference, u32 mask, u32 poll_interval,
                          u64 failed_tests, bool vo_sleep) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    auto* ring = GetThreadRing();
    if (ring == nullptr) {
        return;
    }

    const u32 identity = PackPm4Identity(engine, queue_id, 0x3c, depth);
    u64 hash = NonZeroPm4Hash(location ^ (static_cast<u64>(identity) << 32) ^ control);
    hash = NonZeroPm4Hash(hash ^ (static_cast<u64>(reference) << 32) ^ mask);
    hash = NonZeroPm4Hash(hash ^ poll_interval);
    size_t slot = hash & (Pm4WaitCapacity - 1);
    for (size_t probe = 0; probe < Pm4HashProbeLimit; ++probe) {
        auto& entry = ring->pm4.waits[slot];
        const u64 entry_hash = entry.hash.load(std::memory_order_acquire);
        if (entry_hash == hash && entry.location == location && entry.identity == identity &&
            entry.control == control && entry.reference == reference && entry.mask == mask &&
            entry.poll_interval == poll_interval) {
            AddSingleWriter(entry.packets, 1);
            AddSingleWriter(entry.failed_tests, failed_tests);
            AddSingleWriter(entry.immediate_passes, failed_tests == 0);
            AddSingleWriter(entry.vo_sleeps, vo_sleep);
            return;
        }
        if (entry_hash == 0) {
            entry.location = location;
            entry.identity = identity;
            entry.control = control;
            entry.reference = reference;
            entry.mask = mask;
            entry.poll_interval = poll_interval;
            entry.hash.store(hash, std::memory_order_release);
            AddSingleWriter(entry.packets, 1);
            AddSingleWriter(entry.failed_tests, failed_tests);
            AddSingleWriter(entry.immediate_passes, failed_tests == 0);
            AddSingleWriter(entry.vo_sleeps, vo_sleep);
            return;
        }
        slot = (slot + 1) & (Pm4WaitCapacity - 1);
    }
    AddSingleWriter(ring->pm4.wait_overflow, 1);
}
#endif

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
EventSeq NextEventSeqEnabled() noexcept {
    return g_event_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

PacketSeq NextPacketSeqEnabled() noexcept {
    return g_packet_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

ProducerSeq NextProducerSeqEnabled() noexcept {
    return g_producer_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

FenceSeq NextFenceSeqEnabled() noexcept {
    return g_fence_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

FenceGen NextFenceGenEnabled() noexcept {
    return g_label_generation_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

WaitSeq NextWaitSeqEnabled() noexcept {
    return g_wait_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

ResourceSeq NextResourceSeqEnabled() noexcept {
    return g_resource_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

ResourceVersion NextResourceVersionEnabled() noexcept {
    return g_resource_version.fetch_add(1, std::memory_order_relaxed) + 1;
}

SubmitSeq NextSubmitSeqEnabled() noexcept {
    return g_submit_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

CpuAccessSeq NextCpuAccessSeqEnabled() noexcept {
    return g_cpu_access_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

ReadbackSeq NextReadbackSeqEnabled() noexcept {
    return g_readback_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

CmdBufferSeq NextCmdBufferSeqEnabled() noexcept {
    return g_cmdbuf_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

FrameSeq CurrentFrameSeqEnabled() noexcept {
    return g_frame_sequence;
}

CmdBufferSeq CurrentCmdBufferSeqEnabled() noexcept {
    return g_cmdbuf_sequence.load(std::memory_order_relaxed);
}

PacketSeq CurrentPacketSeqEnabled() noexcept {
    return g_packet_sequence.load(std::memory_order_relaxed);
}

ProducerSeq CurrentProducerSeqEnabled() noexcept {
    return g_producer_sequence.load(std::memory_order_relaxed);
}

void RecordSyncPm4PacketEnabled(const SyncPm4PacketSample& sample) noexcept {
    WriteRelationalSample(g_sync_pm4_records, g_sync_pm4_written, sample);
}

void RecordProducerBeginEnabled(const ProducerBeginSample& sample) noexcept {
    WriteRelationalSample(g_producer_begin_records, g_producer_begin_written, sample);
}

void RecordProducerEndEnabled(const ProducerEndSample& sample) noexcept {
    WriteRelationalSample(g_producer_end_records, g_producer_end_written, sample);
}

void RecordProducerRecordEnabled(const ProducerRecordSample& sample) noexcept {
    WriteRelationalSample(g_producer_full_records, g_producer_full_written, sample);
}

void RecordResourceWriteEnabled(const ResourceWriteSample& sample) noexcept {
    WriteRelationalSample(g_resource_write_records, g_resource_write_written, sample);
    if (tl_producer.active) {
        tl_producer.write_range_count++;
        tl_producer.write_bytes += sample.guest_size;
        tl_producer.write_resource_count++;
        if (sample.write_kind == ResourceWriteKind::StorageImage ||
            sample.write_kind == ResourceWriteKind::StorageBuffer ||
            sample.write_kind == ResourceWriteKind::Transfer) {
            tl_producer.is_promoted = true;
        }
    }
    if (sample.guest_addr != 0) {
        std::scoped_lock lock{g_range_coherence_mutex};
        size_t slot = (sample.guest_addr >> 12) & (RangeCoherenceCapacity - 1);
        for (size_t probe = 0; probe < 16; ++probe) {
            auto& entry = g_range_coherence[slot];
            if (!entry.valid || entry.addr == sample.guest_addr) {
                entry.addr = sample.guest_addr;
                entry.size = sample.guest_size;
                entry.latest_version = sample.version;
                entry.authoritative_owner = sample.resource_type;
                entry.authoritative_resource = sample.resource_id;
                entry.producer_seq = sample.producer_seq;
                entry.valid = true;
                break;
            }
            slot = (slot + 1) & (RangeCoherenceCapacity - 1);
        }
    }
}

void RecordFenceCreateEnabled(const FenceCreateSample& in_sample) noexcept {
    FenceCreateSample sample = in_sample;
    if (sample.generation == 0) {
        sample.generation = NextFenceGenEnabled();
    }
    WriteRelationalSample(g_fence_create_records, g_fence_create_written, sample);
    Add(Counter::FenceTotalCount);

    // Auto-link active epoch write ranges to this fence
    {
        std::scoped_lock lock{g_range_coherence_mutex};
        for (const auto& entry : g_range_coherence) {
            if (entry.valid && entry.latest_version > entry.host_version) {
                RecordFenceEpochLinkEnabled(FenceEpochLinkSample{
                    .fence_seq = sample.fence_seq,
                    .producer_seq = entry.producer_seq,
                    .resource_id = entry.authoritative_resource,
                    .version = entry.latest_version,
                    .guest_addr = entry.addr,
                    .guest_size = entry.size,
                    .resource_kind = entry.authoritative_owner,
                    .write_kind = ResourceWriteKind::StorageImage,
                    .queue_id = sample.queue_id,
                    .stage = 0,
                    .reason = FenceLinkReason::CandidateWriteEpoch,
                });
            }
        }
    }

    // Shadow decision evaluation
    {
        FenceClassification would_class = FenceClassification::GpuOnly;
        WaitConfidence conf = WaitConfidence::High;
        u32 reason_bits = 0;
        if (sample.interrupt_select != 0) {
            would_class = FenceClassification::CpuVisible;
            reason_bits |= static_cast<u32>(FenceEvidence::InterruptRequested);
        }
        RecordShadowFencePolicyEnabled(ShadowFencePolicySample{
            .fence_seq = sample.fence_seq,
            .would_classify = would_class,
            .would_skip_host_readback = (would_class == FenceClassification::GpuOnly),
            .reason_bits = reason_bits,
            .confidence = conf,
        });
    }

    std::scoped_lock lock{g_active_labels_mutex};
    size_t slot = (sample.label_addr >> 2) & (ActiveLabelTableCapacity - 1);
    for (size_t probe = 0; probe < 16; ++probe) {
        auto& entry = g_active_labels[slot];
        if (entry.valid && entry.addr == sample.label_addr) {
            entry.superseded = true;
            auto& dbg = g_fence_candidate_debug[slot];
            dbg.last_superseded_fence = entry.fence_seq;
            dbg.last_supersede_packet = sample.packet_seq;
            dbg.supersede_count++;
            dbg.last_remove_fence = entry.fence_seq;
            dbg.last_remove_packet = sample.packet_seq;
            dbg.last_remove_reason = CandidateRemoveReason::SupersededByNewFence;
            dbg.remove_count++;

            FenceClassification final_class = FenceClassification::Ambiguous;
            WaitConfidence conf = WaitConfidence::Low;
            if (entry.gpu_wait_count > 0 && entry.cpu_label_read_count == 0 &&
                entry.cpu_protected_data_read_count == 0 &&
                (entry.evidence_bits & static_cast<u32>(FenceEvidence::InterruptRequested)) == 0) {
                final_class = FenceClassification::GpuOnly;
                conf = WaitConfidence::High;
                Add(Counter::FenceGpuOnlyCount);
            } else if (entry.cpu_label_read_count > 0 || entry.cpu_protected_data_read_count > 0 ||
                       (entry.evidence_bits & static_cast<u32>(FenceEvidence::InterruptRequested)) != 0) {
                final_class = FenceClassification::CpuVisible;
                conf = WaitConfidence::High;
                Add(Counter::FenceCpuVisibleCount);
            } else {
                Add(Counter::FenceAmbiguousCount);
            }
            RecordFenceClassificationEnabled(FenceClassificationSample{
                .fence_seq = entry.fence_seq,
                .gpu_wait_count = entry.gpu_wait_count,
                .cpu_label_read_count = entry.cpu_label_read_count,
                .cpu_label_write_count = entry.cpu_label_write_count,
                .irq_requested = entry.irq_requested,
                .cpu_protected_data_read_count = entry.cpu_protected_data_read_count,
                .classification_final = final_class,
                .confidence = conf,
                .reason_bits = entry.evidence_bits,
            });
            DisarmLabelReadWatchEnabled(entry.fence_seq, entry.addr, sizeof(u32));
            entry = {};
            break;
        }
        if (!entry.valid) {
            break;
        }
        slot = (slot + 1) & (ActiveLabelTableCapacity - 1);
    }

    slot = (sample.label_addr >> 2) & (ActiveLabelTableCapacity - 1);
    for (size_t probe = 0; probe < 16; ++probe) {
        auto& entry = g_active_labels[slot];
        if (!entry.valid || entry.addr == sample.label_addr) {
            entry.fence_seq = sample.fence_seq;
            entry.generation = sample.generation;
            entry.addr = sample.label_addr;
            entry.value = sample.label_value;
            entry.create_packet = sample.packet_seq;
            entry.queue_id = sample.queue_id;
            entry.gpu_wait_count = 0;
            entry.cpu_label_read_count = 0;
            entry.cpu_label_write_count = 0;
            entry.cpu_protected_data_read_count = 0;
            entry.evidence_bits = sample.classification_reason_bits;
            entry.irq_requested = (sample.interrupt_select != 0);
            entry.superseded = false;
            entry.valid = true;

            auto& dbg = g_fence_candidate_debug[slot];
            dbg.last_insert_fence = sample.fence_seq;
            dbg.last_insert_generation = sample.generation;
            dbg.last_insert_packet = sample.packet_seq;
            dbg.last_value = sample.label_value;
            dbg.last_mask = 0xffffffff;
            dbg.insert_count++;
            break;
        }
        slot = (slot + 1) & (ActiveLabelTableCapacity - 1);
    }
}

void RecordFenceEpochLinkEnabled(const FenceEpochLinkSample& sample) noexcept {
    WriteRelationalSample(g_fence_epoch_link_records, g_fence_epoch_link_written, sample);
    Add(Counter::FenceEpochLinks);
}

void RecordFenceMatchAttemptEnabled(const FenceMatchAttemptSample& sample) noexcept {
    WriteRelationalSample(g_fence_match_records, g_fence_match_written, sample);
}

void RecordFenceMatchDiagnosticEnabled(const FenceMatchDiagnosticSample& sample) noexcept {
    WriteRelationalSample(g_fence_match_diagnostic_records, g_fence_match_diagnostic_written, sample);
    Add(Counter::FenceMatchDiagnosticCount);
}

void RecordResourceEpochPromotedEnabled(const ResourceEpochPromotedSample& sample) noexcept {
    WriteRelationalSample(g_resource_epoch_promoted_records, g_resource_epoch_promoted_written, sample);
    Add(Counter::ResourceEpochPromotedCount);
}

void RecordFenceResourceLinkEnabled(const FenceResourceLinkSample& sample) noexcept {
    WriteRelationalSample(g_fence_resource_link_records, g_fence_resource_link_written, sample);
    Add(Counter::FenceResourceLinkCount);
}

void RecordWaitCreateEnabled(const WaitCreateSample& in_sample) noexcept {
    WaitCreateSample sample = in_sample;
    {
        std::scoped_lock lock{g_active_labels_mutex};
        size_t slot = (sample.wait_addr >> 2) & (ActiveLabelTableCapacity - 1);
        for (size_t probe = 0; probe < 16; ++probe) {
            auto& entry = g_active_labels[slot];
            if (entry.valid && entry.addr == sample.wait_addr) {
                entry.gpu_wait_count++;
                entry.evidence_bits |= static_cast<u32>(FenceEvidence::MatchedGpuWait);
                sample.matched_fence_seq = entry.fence_seq;
                sample.matched_generation = entry.generation;
                sample.matched_fence_confidence = WaitConfidence::High;
                if (sample.packet_seq >= entry.create_packet) {
                    sample.packets_since_fence = static_cast<u32>(sample.packet_seq - entry.create_packet);
                }
                Add(Counter::WaitMatchedFences);
                break;
            }
            if (!entry.valid) {
                break;
            }
            slot = (slot + 1) & (ActiveLabelTableCapacity - 1);
        }
    }
    if (sample.matched_fence_seq == 0) {
        Add(Counter::WaitUnmatchedFences);
    }
    WriteRelationalSample(g_wait_create_records, g_wait_create_written, sample);
}

void RecordWaitCompleteEnabled(const WaitCompleteSample& sample) noexcept {
    WriteRelationalSample(g_wait_complete_records, g_wait_complete_written, sample);
    Add(Counter::WaitCompleteCount);
    if (sample.fence_seq != 0) {
        ArmConsumerProbeEnabled(sample.fence_seq, sample.wait_seq, CurrentPacketSeqEnabled());
    } else if (sample.shadow_fence_seq != 0) {
        ArmConsumerProbeEnabled(sample.shadow_fence_seq, sample.wait_seq, CurrentPacketSeqEnabled(),
                                ConsumerProbeKind::StructuralShadowFence);
    }
}

void RecordFirstConsumerEnabled(const FirstConsumerSample& sample) noexcept {
    WriteRelationalSample(g_first_consumer_records, g_first_consumer_written, sample);
    Add(Counter::FirstConsumerCount);
}

void RecordFenceClassificationEnabled(const FenceClassificationSample& sample) noexcept {
    WriteRelationalSample(g_fence_classification_records, g_fence_classification_written, sample);
}

void RecordCpuMemoryAccessEnabled(const CpuAccessSample& in_sample) noexcept {
    CpuAccessSample sample = in_sample;
    sample.write_origin = tl_memory.origin;
    Add(Counter::CpuAccessCount);
    bool is_stale = false;
    {
        std::scoped_lock lock{g_active_labels_mutex};
        size_t slot = (sample.guest_addr >> 2) & (ActiveLabelTableCapacity - 1);
        for (size_t probe = 0; probe < 16; ++probe) {
            auto& entry = g_active_labels[slot];
            if (entry.valid && entry.addr <= sample.guest_addr && sample.guest_addr < entry.addr + 8) {
                sample.is_label_range = true;
                sample.matched_fence_seq = entry.fence_seq;
                if (sample.access_type == CpuAccessType::Read) {
                    entry.cpu_label_read_count++;
                    entry.evidence_bits |= static_cast<u32>(FenceEvidence::CpuReadObserved);
                } else {
                    entry.cpu_label_write_count++;
                    entry.evidence_bits |= static_cast<u32>(FenceEvidence::CpuWriteObserved);
                }
                RecordCpuLabelAccessEnabled(CpuLabelAccessSample{
                    .fence_seq = entry.fence_seq,
                    .generation = entry.generation,
                    .access_type = sample.access_type,
                    .guest_addr = sample.guest_addr,
                    .timestamp_ns = Timestamp(),
                    .guest_thread_id = sample.thread_id,
                });
                break;
            }
            if (!entry.valid) {
                break;
            }
            slot = (slot + 1) & (ActiveLabelTableCapacity - 1);
        }
    }
    {
        std::scoped_lock lock{g_range_coherence_mutex};
        size_t slot = (sample.guest_addr >> 12) & (RangeCoherenceCapacity - 1);
        for (size_t probe = 0; probe < 16; ++probe) {
            auto& entry = g_range_coherence[slot];
            if (entry.valid && entry.addr <= sample.guest_addr && sample.guest_addr < entry.addr + entry.size) {
                sample.latest_version = entry.latest_version;
                sample.host_version = entry.host_version;
                sample.authoritative_owner = entry.authoritative_owner;
                sample.authoritative_resource = entry.authoritative_resource;
                sample.authoritative_tick = entry.authoritative_tick;
                if (entry.latest_version > entry.host_version) {
                    is_stale = true;
                    Add(Counter::CpuStaleAccessCount);
                }
                break;
            }
            if (!entry.valid) {
                break;
            }
            slot = (slot + 1) & (RangeCoherenceCapacity - 1);
        }
    }
    WriteRelationalSample(g_cpu_access_records, g_cpu_access_written, sample);
    if (is_stale && sample.access_type == CpuAccessType::Read) {
        RecordCpuReadRequiresMaterializationEnabled(CpuMaterializationSample{
            .cpu_access_seq = sample.cpu_access_seq,
            .guest_addr = sample.guest_addr,
            .size = sample.size,
            .latest_version = sample.latest_version,
            .host_version = sample.host_version,
            .producer_seq = 0,
            .producer_tick = sample.authoritative_tick,
            .source_resource_id = sample.authoritative_resource,
            .source_resource_type = sample.authoritative_owner,
            .readback_already_scheduled = false,
            .readback_ready = false,
        });
        Add(Counter::CpuMaterializationCount);
    }
}

void RecordCpuLabelAccessEnabled(const CpuLabelAccessSample& sample) noexcept {
    WriteRelationalSample(g_cpu_label_records, g_cpu_label_written, sample);
    Add(Counter::CpuLabelAccessCount);
}

void RecordCpuReadRequiresMaterializationEnabled(const CpuMaterializationSample& sample) noexcept {
    WriteRelationalSample(g_cpu_materialization_records, g_cpu_materialization_written, sample);
}

void RecordStaleGuestSourceAttemptEnabled(const StaleGuestAttemptSample& sample) noexcept {
    WriteRelationalSample(g_stale_guest_records, g_stale_guest_written, sample);
    Add(Counter::StaleGuestAttempts);
}

void RecordGpuAliasMaterializeEnabled(const GpuAliasMaterializeSample& sample) noexcept {
    WriteRelationalSample(g_gpu_alias_records, g_gpu_alias_written, sample);
    Add(Counter::GpuAliasMaterializations);
}

void RecordReadbackScheduleEnabled(const ReadbackScheduleSample& sample) noexcept {
    WriteRelationalSample(g_readback_schedule_records, g_readback_schedule_written, sample);
    Add(Counter::ReadbackScheduleCount);
}

void RecordReadbackSubmitEnabled(const ReadbackSubmitSample& sample) noexcept {
    WriteRelationalSample(g_readback_submit_records, g_readback_submit_written, sample);
    Add(Counter::ReadbackSubmitCount);
}

void RecordReadbackReadyEnabled(const ReadbackReadySample& sample) noexcept {
    WriteRelationalSample(g_readback_ready_records, g_readback_ready_written, sample);
    Add(Counter::ReadbackReadyCount);
}

void RecordReadbackCommitEnabled(const ReadbackCommitSample& sample) noexcept {
    WriteRelationalSample(g_readback_commit_records, g_readback_commit_written, sample);
    Add(Counter::ReadbackCommitCount);
    if (sample.guest_addr != 0) {
        std::scoped_lock lock{g_range_coherence_mutex};
        size_t slot = (sample.guest_addr >> 12) & (RangeCoherenceCapacity - 1);
        for (size_t probe = 0; probe < 16; ++probe) {
            auto& entry = g_range_coherence[slot];
            if (entry.valid && entry.addr == sample.guest_addr) {
                entry.host_version = sample.version;
                break;
            }
            if (!entry.valid) {
                break;
            }
            slot = (slot + 1) & (RangeCoherenceCapacity - 1);
        }
    }
}

void RecordFenceSignalEnabled(const FenceSignalSample& sample) noexcept {
    WriteRelationalSample(g_fence_signal_records, g_fence_signal_written, sample);
    Add(Counter::FenceSignalCount);
}

void RecordHostWaitEnabled(const HostWaitSample& sample) noexcept {
    WriteRelationalSample(g_host_wait_records, g_host_wait_written, sample);
    Add(Counter::HostWaitCount);
    if (sample.reason == HostWaitReason::SchedulerFinish) {
        Add(Counter::HostWaitFinishCount);
    } else if (sample.reason == HostWaitReason::FenceCpuVisibility) {
        Add(Counter::HostWaitCpuVisibilityCount);
    } else if (sample.reason == HostWaitReason::WaitRegMemProgress) {
        Add(Counter::HostWaitWaitProgressCount);
    }
}

void RecordSubmitRecordEnabled(const SubmitRecordSample& sample) noexcept {
    WriteRelationalSample(g_submit_records, g_submit_written, sample);
}

void RecordShadowFencePolicyEnabled(const ShadowFencePolicySample& sample) noexcept {
    WriteRelationalSample(g_shadow_fence_records, g_shadow_fence_written, sample);
}

void RecordTraceGapEnabled(const TraceGapSample& sample) noexcept {
    WriteRelationalSample(g_trace_gap_records, g_trace_gap_written, sample);
    Add(Counter::TraceGapCount);
}

void RecordRingHealthEnabled(const RingHealthSample& sample) noexcept {
    WriteRelationalSample(g_ring_health_records, g_ring_health_written, sample);
    Add(Counter::RingHealthCount);
}

void RegisterSubmitTickEnabled(u64 signal_tick, SubmitSeq submit_seq) noexcept {
    if (signal_tick == 0) return;
    std::scoped_lock lock{g_submit_tick_mutex};
    const size_t slot = (signal_tick ^ (signal_tick >> 12)) & (SubmitTickMapCapacity - 1);
    g_submit_tick_map[slot] = {signal_tick, submit_seq};
}

SubmitSeq LookupSubmitSeqEnabled(u64 signal_tick) noexcept {
    if (signal_tick == 0) return 0;
    std::scoped_lock lock{g_submit_tick_mutex};
    const size_t slot = (signal_tick ^ (signal_tick >> 12)) & (SubmitTickMapCapacity - 1);
    if (g_submit_tick_map[slot].signal_tick == signal_tick) {
        return g_submit_tick_map[slot].submit_seq;
    }
    for (size_t i = 1; i < 8; ++i) {
        const size_t idx = (slot + i) & (SubmitTickMapCapacity - 1);
        if (g_submit_tick_map[idx].signal_tick == signal_tick) {
            return g_submit_tick_map[idx].submit_seq;
        }
    }
    return 0;
}

WatchSeq NextWatchSeqEnabled() noexcept {
    return g_watch_seq.fetch_add(1, std::memory_order_relaxed);
}

void RecordReadbackSourceTerminalEnabled(const ReadbackSourceTerminalSample& sample) noexcept {
    WriteRelationalSample(g_readback_source_terminal_records, g_readback_source_terminal_written, sample);
}

void RecordGuestSourceConsumeEnabled(const GuestSourceConsumeSample& sample) noexcept {
    WriteRelationalSample(g_guest_source_consume_records, g_guest_source_consume_written, sample);
}

void RecordResourceLineageEnabled(const ResourceLineageSample& sample) noexcept {
    WriteRelationalSample(g_resource_lineage_records, g_resource_lineage_written, sample);
    std::scoped_lock lock{g_source_watches_mutex};
    auto it = g_active_source_watches.find({sample.source_resource_id, sample.source_resource_version});
    if (it != g_active_source_watches.end()) {
        auto watch = it->second;
        g_active_source_watches.erase(it);
        RecordReadbackSourceTerminalEnabled(ReadbackSourceTerminalSample{
            .watch_seq = watch.watch_seq,
            .fence_seq = watch.fence_seq,
            .readback_seq = watch.readback_seq,
            .source_resource_id = sample.source_resource_id,
            .source_resource_version = sample.source_resource_version,
            .terminal_kind = TerminalKind::GpuAliasMaterialization,
            .consumer_producer_seq = watch.producer_seq,
            .consumer_packet_seq = watch.producer_packet,
            .consumer_resource_id = sample.destination_resource_id,
            .consumer_resource_version = sample.destination_resource_version,
            .access_path = ConsumerAccessPath::ImageAliasResolve,
            .overlap_addr = watch.guest_addr,
            .overlap_size = watch.size,
            .packets_since_fence = 0,
            .ns_since_fence = 0,
            .correlation_status = CorrelationStatus::Complete,
            .cmd_buffer_seq = 0,
            .submit_seq = 0,
        });
        watch.resource_id = sample.destination_resource_id;
        watch.resource_version = sample.destination_resource_version;
        watch.watch_seq = NextWatchSeqEnabled();
        g_active_source_watches[{watch.resource_id, watch.resource_version}] = watch;
    }
}

void RecordCpuReadObservationEnabled(const CpuReadObservationSample& sample) noexcept {
    WriteRelationalSample(g_cpu_read_observation_records, g_cpu_read_observation_written, sample);
}

void RecordSemanticReadFaultEnabled(const SemanticReadFaultSample& sample) noexcept {
    WriteRelationalSample(g_semantic_read_fault_records, g_semantic_read_fault_written, sample);
}

void RecordSemanticReadUnknownEnabled(const SemanticReadUnknownSample& sample) noexcept {
    WriteRelationalSample(g_semantic_read_unknown_records, g_semantic_read_unknown_written, sample);
}

void RecordResourceBarrierLinkEnabled(const ResourceBarrierLinkSample& sample) noexcept {
    WriteRelationalSample(g_resource_barrier_link_records, g_resource_barrier_link_written, sample);
}

void RecordAcquireMemEnabled(const AcquireMemSample& sample) noexcept {
    WriteRelationalSample(g_acquire_mem_records, g_acquire_mem_written, sample);
    Add(Counter::AcquireMemCalls);
}

void RecordFastpathCandidateEnabled(const FastpathCandidateSample& sample) noexcept {
    WriteRelationalSample(g_fastpath_candidate_records, g_fastpath_candidate_written, sample);
}

void RecordGpuAuthorityCreateEnabled(const GpuAuthorityCreateSample& sample) noexcept {
    WriteRelationalSample(g_gpu_authority_create_records, g_gpu_authority_create_written, sample);
}

void RecordVirtualFenceCreateEnabled(const VirtualFenceCreateSample& sample) noexcept {
    WriteRelationalSample(g_virtual_fence_create_records, g_virtual_fence_create_written, sample);
}

void RecordVirtualWaitConsumeEnabled(const VirtualWaitConsumeSample& sample) noexcept {
    WriteRelationalSample(g_virtual_wait_consume_records, g_virtual_wait_consume_written, sample);
}

void RecordAsyncLabelSignalEnabled(const AsyncLabelSignalSample& sample) noexcept {
    WriteRelationalSample(g_async_label_signal_records, g_async_label_signal_written, sample);
}

void RecordAuthorityGpuConsumeEnabled(const AuthorityGpuConsumeSample& sample) noexcept {
    WriteRelationalSample(g_authority_gpu_consume_records, g_authority_gpu_consume_written, sample);
}

void RecordAuthorityBarrierValidationEnabled(const AuthorityBarrierValidationSample& sample) noexcept {
    WriteRelationalSample(g_authority_barrier_validation_records, g_authority_barrier_validation_written, sample);
}

void RecordAuthorityRamDemandEnabled(const AuthorityRamDemandSample& sample) noexcept {
    WriteRelationalSample(g_authority_ram_demand_records, g_authority_ram_demand_written, sample);
}

void RecordLazyMaterializeBeginEnabled(const LazyMaterializeBeginSample& sample) noexcept {
    WriteRelationalSample(g_lazy_materialize_begin_records, g_lazy_materialize_begin_written, sample);
}

void RecordLazyMaterializeEndEnabled(const LazyMaterializeEndSample& sample) noexcept {
    WriteRelationalSample(g_lazy_materialize_end_records, g_lazy_materialize_end_written, sample);
}

void RecordAuthorityRamConsumeEnabled(const AuthorityRamConsumeSample& sample) noexcept {
    WriteRelationalSample(g_authority_ram_consume_records, g_authority_ram_consume_written, sample);
}

void RecordAuthorityCpuReadEnabled(const AuthorityCpuReadSample& sample) noexcept {
    WriteRelationalSample(g_authority_cpu_read_records, g_authority_cpu_read_written, sample);
}

void RecordAuthoritySupersedeEnabled(const AuthoritySupersedeSample& sample) noexcept {
    WriteRelationalSample(g_authority_supersede_records, g_authority_supersede_written, sample);
}

void RecordFastpathFallbackEnabled(const FastpathFallbackSample& sample) noexcept {
    WriteRelationalSample(g_fastpath_fallback_records, g_fastpath_fallback_written, sample);
}

void RecordConservativeDownloadDecisionEnabled(const ConservativeDownloadDecisionSample& sample) noexcept {
    WriteRelationalSample(g_conservative_download_decision_records, g_conservative_download_decision_written, sample);
}

void RecordAuthorityConservativeReadbackSuppressedEnabled(const AuthorityConservativeReadbackSuppressedSample& sample) noexcept {
    WriteRelationalSample(g_authority_conservative_readback_suppressed_records, g_authority_conservative_readback_suppressed_written, sample);
}

void RecordAuthorityHostMaterializeRequiredEnabled(const AuthorityHostMaterializeRequiredSample& sample) noexcept {
    WriteRelationalSample(g_authority_host_materialize_required_records, g_authority_host_materialize_required_written, sample);
}

void RecordFastpathWaitDecisionEnabled(const FastpathWaitDecisionSample& sample) noexcept {
    WriteRelationalSample(g_fastpath_wait_decision_records, g_fastpath_wait_decision_written, sample);
}

void RecordVirtualFenceForcedCompletionEnabled(const VirtualFenceForcedCompletionSample& sample) noexcept {
    WriteRelationalSample(g_virtual_fence_forced_completion_records, g_virtual_fence_forced_completion_written, sample);
}

void RecordCpuToGpuLabelWaitEnabled(const CpuToGpuLabelWaitSample& sample) noexcept {
    if (sample.last_guest_write_ts == 0) {
        std::scoped_lock lk{g_guest_label_writes_mutex};
        auto it = g_guest_label_writes.find(sample.label_addr);
        if (it != g_guest_label_writes.end()) {
            auto copy = sample;
            copy.last_guest_write_ts = it->second.timestamp_ns;
            copy.last_guest_write_value = it->second.value;
            copy.writer_thread_id = it->second.thread_id;
            if (copy.wait_end >= copy.last_guest_write_ts) {
                copy.delta_write_to_wait_complete_ns = copy.wait_end - copy.last_guest_write_ts;
            }
            WriteRelationalSample(g_cpu_to_gpu_label_wait_records, g_cpu_to_gpu_label_wait_written, copy);
            return;
        }
    }
    WriteRelationalSample(g_cpu_to_gpu_label_wait_records, g_cpu_to_gpu_label_wait_written, sample);
}

void RecordGuestCpuLabelWriteEnabled(VAddr addr, u32 val, u64 timestamp, u64 thread_id) noexcept {
    std::scoped_lock lk{g_guest_label_writes_mutex};
    g_guest_label_writes[addr] = LastGuestWriteInfo{
        .timestamp_ns = timestamp != 0 ? timestamp : Timestamp(),
        .value = val,
        .thread_id = thread_id,
    };
}

u64 NextCandidateSeqEnabled() noexcept {
    return g_candidate_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

u64 NextAuthoritySeqEnabled() noexcept {
    return g_authority_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

u64 NextVirtualFenceSeqEnabled() noexcept {
    return g_virtual_fence_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

u64 NextRamDemandSeqEnabled() noexcept {
    return g_ram_demand_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

u64 NextRamDemandGroupSeqEnabled() noexcept {
    return g_ram_demand_group_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

u64 NextMaterializeSeqEnabled() noexcept {
    return g_materialize_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

u64 NextConsumerSeqEnabled() noexcept {
    return g_consumer_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

void ArmReadbackSourceWatchEnabled(const ReadbackSourceWatch& watch) noexcept {
    std::scoped_lock lock{g_source_watches_mutex};
    g_active_source_watches[{watch.resource_id, watch.resource_version}] = watch;
}

bool HasActiveReadbackSourceWatchEnabled(ResourceId res_id, ResourceVersion ver) noexcept {
    std::scoped_lock lock{g_source_watches_mutex};
    return g_active_source_watches.contains({res_id, ver});
}

ReadbackSourceWatch GetActiveReadbackSourceWatchEnabled(ResourceId res_id, ResourceVersion ver) noexcept {
    std::scoped_lock lock{g_source_watches_mutex};
    auto it = g_active_source_watches.find({res_id, ver});
    return it != g_active_source_watches.end() ? it->second : ReadbackSourceWatch{};
}

void ResolveReadbackSourceWatchEnabled(ResourceId res_id, ResourceVersion ver, VAddr addr, u64 size,
                                       TerminalKind kind, ProducerSeq consumer_prod,
                                       PacketSeq consumer_pkt, ResourceId consumer_res,
                                       ResourceVersion consumer_ver,
                                       ConsumerAccessPath path,
                                       CmdBufferSeq cmd_buf, SubmitSeq submit_seq) noexcept {
    ReadbackSourceWatch matched_watch{};
    bool found = false;
    ActiveSourceWatchKey matched_key{};
    {
        std::scoped_lock lock{g_source_watches_mutex};
        if (res_id != 0 && ver != 0) {
            auto it = g_active_source_watches.find({res_id, ver});
            if (it != g_active_source_watches.end()) {
                matched_watch = it->second;
                matched_key = it->first;
                found = true;
            }
        }
        if (!found && addr != 0 && size != 0) {
            for (auto it = g_active_source_watches.begin(); it != g_active_source_watches.end(); ++it) {
                const auto& w = it->second;
                if (w.guest_addr != 0 && w.size != 0) {
                    if (std::max(addr, w.guest_addr) < std::min(addr + size, w.guest_addr + w.size)) {
                        matched_watch = w;
                        matched_key = it->first;
                        found = true;
                        break;
                    }
                }
            }
        }
        if (found) {
            g_active_source_watches.erase(matched_key);
        }
    }

    if (found) {
        const u64 now_ns = Timestamp();
        const u64 ns_since = now_ns > matched_watch.create_timestamp_ns ? now_ns - matched_watch.create_timestamp_ns : 0;
        const u32 pkts_since = consumer_pkt > matched_watch.fence_packet ? static_cast<u32>(consumer_pkt - matched_watch.fence_packet) : 0;
        SubmitSeq resolved_submit = submit_seq;
        if (resolved_submit == 0 && cmd_buf != 0) {
            resolved_submit = LookupCmdBufferSubmitEnabled(cmd_buf);
        }

        RecordReadbackSourceTerminalEnabled(ReadbackSourceTerminalSample{
            .watch_seq = matched_watch.watch_seq,
            .fence_seq = matched_watch.fence_seq,
            .readback_seq = matched_watch.readback_seq,
            .source_resource_id = matched_watch.resource_id,
            .source_resource_version = matched_watch.resource_version,
            .terminal_kind = kind,
            .consumer_producer_seq = consumer_prod,
            .consumer_packet_seq = consumer_pkt,
            .consumer_resource_id = consumer_res,
            .consumer_resource_version = consumer_ver,
            .access_path = path,
            .overlap_addr = addr != 0 ? addr : matched_watch.guest_addr,
            .overlap_size = size != 0 ? size : matched_watch.size,
            .packets_since_fence = pkts_since,
            .ns_since_fence = ns_since,
            .correlation_status = CorrelationStatus::Complete,
            .cmd_buffer_seq = cmd_buf,
            .submit_seq = resolved_submit,
        });
    }
}

void UpdateHostVersionEnabled(VAddr addr, u64 size, ResourceVersion ver, HostVersionOrigin origin,
                              ReadbackSeq readback_seq, ResourceId res_id) noexcept {
    if (addr == 0 || size == 0) return;
    std::scoped_lock lock{g_host_version_mutex};
    const VAddr start = addr;
    const VAddr end = addr + size;
    std::erase_if(g_host_version_ranges, [start, end](const HostVersionRange& r) {
        return std::max(start, r.start) < std::min(end, r.end);
    });
    if (g_host_version_ranges.size() > 8192) {
        g_host_version_ranges.erase(g_host_version_ranges.begin(), g_host_version_ranges.begin() + 1024);
    }
    g_host_version_ranges.push_back(HostVersionRange{
        .start = start,
        .end = end,
        .state = {
            .version = ver,
            .origin = origin,
            .origin_readback_seq = readback_seq,
            .origin_resource_id = res_id,
            .origin_event_seq = NextEventSeqEnabled(),
        },
    });
}

void CheckGuestSourceConsumeEnabled(VAddr addr, u64 size, ProducerSeq prod_seq, PacketSeq pkt_seq,
                                    ResourceType dst_kind, ResourceId dst_id,
                                    GuestSourceConsumePath path) noexcept {
    if (addr == 0 || size == 0) return;
    ReadbackSeq origin_rb = 0;
    ResourceId origin_res = 0;
    ResourceVersion origin_ver = 0;
    bool found = false;
    {
        std::scoped_lock lock{g_host_version_mutex};
        const VAddr start = addr;
        const VAddr end = addr + size;
        for (const auto& r : g_host_version_ranges) {
            if (r.state.origin == HostVersionOrigin::ReadbackCommit &&
                std::max(start, r.start) < std::min(end, r.end)) {
                origin_rb = r.state.origin_readback_seq;
                origin_res = r.state.origin_resource_id;
                origin_ver = r.state.version;
                found = true;
                break;
            }
        }
    }
    if (found) {
        RecordGuestSourceConsumeEnabled(GuestSourceConsumeSample{
            .origin_readback_seq = origin_rb,
            .origin_resource_id = origin_res,
            .origin_resource_version = origin_ver,
            .guest_addr = addr,
            .size = size,
            .consumer_producer_seq = prod_seq,
            .consumer_packet_seq = pkt_seq,
            .destination_kind = dst_kind,
            .destination_resource_id = dst_id,
            .path = path,
        });
        ResolveReadbackSourceWatchEnabled(origin_res, origin_ver, addr, size,
                                          TerminalKind::GuestRamGpuUpload, prod_seq, pkt_seq,
                                          dst_id, 0, ConsumerAccessPath::GuestRamUpload);
    }
}

void ArmReadWatchInterestEnabled(const ReadWatchInterest& interest) noexcept {
    if (interest.guest_addr == 0 || interest.size == 0) return;
    const VAddr start_page = interest.guest_addr & ~0xFFFULL;
    const VAddr end_page = (interest.guest_addr + interest.size - 1) & ~0xFFFULL;
    const u64 num_pages = (end_page - start_page) / 4096 + 1;

    if (interest.kind == ReadWatchKind::Label) {
        Add(Counter::SemanticLabelWatchArms);
        Add(Counter::SemanticLabelWatchPages, num_pages);
        Add(Counter::SemanticActiveLabelWatches);
    } else {
        Add(Counter::SemanticDataWatchArms);
        Add(Counter::SemanticDataWatchPages, num_pages);
        Add(Counter::SemanticActiveDataWatches);
    }

    std::scoped_lock lock{g_read_interests_mutex};
    for (VAddr page = start_page; page <= end_page; page += 4096) {
        g_read_watch_interests[page].push_back(interest);
    }
}

void DisarmLabelReadWatchEnabled(FenceSeq fence_seq, VAddr guest_addr, u64 size,
                                 SemanticWatchCancelReason reason,
                                 DisarmWatchCallback disarm_cb, void* user_data) noexcept {
    if (guest_addr == 0 || size == 0) return;
    std::scoped_lock lock{g_read_interests_mutex};
    const VAddr start_page = guest_addr & ~0xFFFULL;
    const VAddr end_page = (guest_addr + size - 1) & ~0xFFFULL;
    for (VAddr page = start_page; page <= end_page; page += 4096) {
        auto it = g_read_watch_interests.find(page);
        if (it == g_read_watch_interests.end()) continue;
        auto& list = it->second;
        for (auto list_it = list.begin(); list_it != list.end(); ) {
            if (list_it->kind == ReadWatchKind::Label && (fence_seq == 0 || list_it->fence_seq == fence_seq) &&
                std::max(guest_addr, list_it->guest_addr) < std::min(guest_addr + size, list_it->guest_addr + list_it->size)) {
                RecordSemanticWatchCancelEnabled(SemanticWatchCancelSample{
                    .fence_seq = list_it->fence_seq,
                    .readback_seq = list_it->readback_seq,
                    .resource_id = list_it->resource_id,
                    .resource_version = list_it->resource_version,
                    .page = page,
                    .guest_addr = list_it->guest_addr,
                    .size = list_it->size,
                    .watch_kind = list_it->kind,
                    .reason = reason,
                    .remaining_page_owners = static_cast<u32>(list.size() - 1),
                });
                switch (reason) {
                case SemanticWatchCancelReason::WaitCompleted:
                    Add(Counter::SemanticWatchCancelWaitComplete);
                    break;
                case SemanticWatchCancelReason::Overwritten:
                    Add(Counter::SemanticWatchCancelOverwritten);
                    break;
                case SemanticWatchCancelReason::PageConflictWrite:
                    Add(Counter::SemanticWatchCancelPageConflict);
                    break;
                case SemanticWatchCancelReason::LivelockBreak:
                    Add(Counter::SemanticFaultLivelockBreaks);
                    break;
                default:
                    break;
                }
                Subtract(Counter::SemanticActiveLabelWatches);
                if (disarm_cb) {
                    disarm_cb(user_data, list_it->guest_addr, list_it->size);
                }
                list_it = list.erase(list_it);
            } else {
                ++list_it;
            }
        }
        if (list.empty()) {
            g_read_watch_interests.erase(it);
        }
    }
}

void SubtractEnabled(Counter counter, u64 value) noexcept {
    if (g_dumping.load(std::memory_order_relaxed)) {
        return;
    }
    if (auto* ring = GetThreadRing()) {
        auto& ctr = ring->counters[static_cast<size_t>(counter)];
        const u64 old = ctr.load(std::memory_order_relaxed);
        ctr.store(old >= value ? old - value : 0, std::memory_order_relaxed);
    }
}

void RecordSemanticWatchCancelEnabled(const SemanticWatchCancelSample& sample) noexcept {
    WriteRelationalSample(g_semantic_watch_cancel_records, g_semantic_watch_cancel_written, sample);
}

void RecordSemanticPageConflictWriteEnabled(const SemanticPageConflictWriteSample& sample) noexcept {
    WriteRelationalSample(g_semantic_page_conflict_records, g_semantic_page_conflict_written, sample);
}

void HandleWriteFaultOnWatchedPageEnabled(VAddr addr, u64 size, u32 thread_id, VAddr rip,
                                         DisarmWatchCallback disarm_cb, void* user_data) noexcept {
    if (addr == 0 || size == 0) return;
    std::scoped_lock lock{g_read_interests_mutex};
    const VAddr page = addr & ~0xFFFULL;
    auto it = g_read_watch_interests.find(page);
    if (it == g_read_watch_interests.end()) return;
    auto& list = it->second;
    for (auto list_it = list.begin(); list_it != list.end(); ) {
        const auto& item = *list_it;
        const VAddr write_end = addr + size;
        const VAddr watch_end = item.guest_addr + item.size;
        const bool overlaps = std::max(addr, item.guest_addr) < std::min(write_end, watch_end);
        SemanticWatchCancelReason cancel_reason;
        if (overlaps) {
            cancel_reason = SemanticWatchCancelReason::Overwritten;
            Add(Counter::SemanticWatchCancelOverwritten);
        } else {
            cancel_reason = SemanticWatchCancelReason::PageConflictWrite;
            Add(Counter::SemanticWatchCancelPageConflict);
            Add(Counter::SemanticPageConflictWrites);
            RecordSemanticPageConflictWriteEnabled(SemanticPageConflictWriteSample{
                .thread_id = thread_id,
                .rip = rip,
                .fault_addr = addr,
                .write_size = size,
                .watched_addr = item.guest_addr,
                .watched_size = item.size,
                .watch_kind = item.kind,
                .fence_seq = item.fence_seq,
                .readback_seq = item.readback_seq,
                .resource_id = item.resource_id,
                .resource_version = item.resource_version,
            });
        }
        RecordSemanticWatchCancelEnabled(SemanticWatchCancelSample{
            .fence_seq = item.fence_seq,
            .readback_seq = item.readback_seq,
            .resource_id = item.resource_id,
            .resource_version = item.resource_version,
            .page = page,
            .guest_addr = item.guest_addr,
            .size = item.size,
            .watch_kind = item.kind,
            .reason = cancel_reason,
            .remaining_page_owners = static_cast<u32>(list.size() - 1),
        });
        if (item.kind == ReadWatchKind::Label) {
            Subtract(Counter::SemanticActiveLabelWatches);
        } else {
            Subtract(Counter::SemanticActiveDataWatches);
        }
        if (disarm_cb) {
            disarm_cb(user_data, item.guest_addr, item.size);
        }
        list_it = list.erase(list_it);
    }
    if (list.empty()) {
        g_read_watch_interests.erase(it);
    }
}

bool CheckCpuReadObservationEnabled(VAddr addr, u64 size, u32 thread_id, VAddr rip,
                                    DisarmWatchCallback disarm_cb, void* user_data) noexcept {
    if (addr == 0 || size == 0) return false;

    // 1. Classify origin according to strict precedence rules (Sections 1, 6, 7, 8):
    // Rule 1: Explicit TLS origin scope
    // Rule 2: RIP in registered guest executable range -> GuestDirect
    // Rule 3: Thread marked as guest execution thread -> GuestHle
    // Rule 4: Fallback -> UnknownHost
    SemanticReadOrigin origin = t_semantic_read_context.origin;
    if (origin == SemanticReadOrigin::None) {
        if (GuestExecutableRegistry::IsGuestRip(rip)) {
            origin = SemanticReadOrigin::GuestDirect;
        } else if (t_is_guest_execution_thread) {
            origin = SemanticReadOrigin::GuestHle;
        } else {
            origin = SemanticReadOrigin::UnknownHost;
        }
    }

    std::scoped_lock lock{g_read_interests_mutex};
    const VAddr start_page = addr & ~0xFFFULL;
    const VAddr end_page = (addr + (size > 0 ? size - 1 : 0)) & ~0xFFFULL;
    bool observed = false;

    for (VAddr page = start_page; page <= end_page; page += 4096) {
        auto it = g_read_watch_interests.find(page);
        if (it == g_read_watch_interests.end()) continue;
        auto& list = it->second;
        for (auto list_it = list.begin(); list_it != list.end(); ) {
            const auto& item = *list_it;
            // Section 10: Fault must overlap exact interest guest range
            if (std::max(addr, item.guest_addr) < std::min(addr + size, item.guest_addr + item.size)) {
                // Lookup current HostVersionState for range (Section 11)
                ResourceVersion cur_host_version = 0;
                HostVersionOrigin cur_host_origin = HostVersionOrigin::Unknown;
                ReadbackSeq cur_host_readback_seq = 0;
                ResourceId cur_host_resource_id = 0;
                bool is_current_version = true;
                {
                    std::scoped_lock hv_lock{g_host_version_mutex};
                    for (const auto& r : g_host_version_ranges) {
                        if (std::max(item.guest_addr, r.start) < std::min(item.guest_addr + item.size, r.end)) {
                            cur_host_version = r.state.version;
                            cur_host_origin = r.state.origin;
                            cur_host_readback_seq = r.state.origin_readback_seq;
                            cur_host_resource_id = r.state.origin_resource_id;
                            if (item.kind == ReadWatchKind::Data && item.resource_id != 0) {
                                if (item.resource_id != cur_host_resource_id || item.resource_version != cur_host_version) {
                                    is_current_version = false;
                                }
                            }
                            break;
                        }
                    }
                }

                // Counters for all faults hitting an active interest (Section 13)
                Add(Counter::SemanticReadFaultsTotal);
                switch (origin) {
                case SemanticReadOrigin::GuestDirect:
                    Add(Counter::SemanticReadGuestDirect);
                    break;
                case SemanticReadOrigin::GuestHle:
                    Add(Counter::SemanticReadGuestHle);
                    break;
                case SemanticReadOrigin::WaitRegMemPoll:
                    Add(Counter::SemanticReadWaitRegMemPoll);
                    break;
                case SemanticReadOrigin::GpuUploadFromGuestRam:
                    Add(Counter::SemanticReadGpuUpload);
                    break;
                case SemanticReadOrigin::RendererInternal:
                    Add(Counter::SemanticReadRendererInternal);
                    break;
                case SemanticReadOrigin::MemoryTrackerInternal:
                    Add(Counter::SemanticReadMemoryTrackerInternal);
                    break;
                case SemanticReadOrigin::TelemetryInternal:
                    Add(Counter::SemanticReadTelemetryInternal);
                    break;
                case SemanticReadOrigin::UnknownHost:
                default:
                    Add(Counter::SemanticReadUnknownHost);
                    break;
                }

                // Low-level audit sample (Section 12)
                RecordSemanticReadFaultEnabled(SemanticReadFaultSample{
                    .thread_id = thread_id,
                    .rip = rip,
                    .fault_addr = addr,
                    .access_size = size,
                    .origin = origin,
                    .watch_kind = item.kind,
                    .fence_seq = item.fence_seq,
                    .readback_seq = item.readback_seq,
                    .resource_id = item.resource_id,
                    .resource_version = item.resource_version,
                    .host_version = cur_host_version,
                    .host_version_origin = cur_host_origin,
                });

                if (origin == SemanticReadOrigin::GuestDirect || origin == SemanticReadOrigin::GuestHle) {
                    if (is_current_version) {
                        const u64 now = Timestamp();
                        const u64 ns_since = now > item.arm_timestamp_ns ? now - item.arm_timestamp_ns : 0;
                        const PacketSeq cur_pkt = CurrentPacketSeqEnabled();
                        const u32 pkts_since = cur_pkt > item.fence_packet ? static_cast<u32>(cur_pkt - item.fence_packet) : 0;

                        if (item.kind == ReadWatchKind::Label) {
                            Add(Counter::SemanticLabelGuestReads);
                        } else {
                            Add(Counter::SemanticDataGuestReads);
                        }
                        Add(Counter::CpuReadObservations);

                        RecordCpuReadObservationEnabled(CpuReadObservationSample{
                            .thread_id = thread_id,
                            .watch_kind = item.kind,
                            .fence_seq = item.fence_seq,
                            .generation = item.generation,
                            .readback_seq = item.readback_seq,
                            .resource_id = item.resource_id,
                            .resource_version = item.resource_version,
                            .fault_addr = addr,
                            .guest_range_addr = item.guest_addr,
                            .guest_range_size = item.size,
                            .ns_since_fence = ns_since,
                            .packets_since_fence = pkts_since,
                        });

                        if (item.resource_id != 0 && item.resource_version != 0) {
                            ResolveReadbackSourceWatchEnabled(item.resource_id, item.resource_version,
                                                              item.guest_addr, item.size,
                                                              TerminalKind::CpuRead, 0, cur_pkt, 0, 0,
                                                              ConsumerAccessPath::CpuFaultRead);
                        }
                        observed = true;
                    }
                    if (disarm_cb) {
                        disarm_cb(user_data, item.guest_addr, item.size);
                    }
                    list_it = list.erase(list_it);
                } else if (origin == SemanticReadOrigin::UnknownHost) {
                    Add(Counter::SemanticPrecisionUnknownReads);
                    RecordSemanticReadUnknownEnabled(SemanticReadUnknownSample{
                        .thread_id = thread_id,
                        .rip = rip,
                        .fault_addr = addr,
                        .access_size = size,
                        .watch_kind = item.kind,
                        .fence_seq = item.fence_seq,
                        .readback_seq = item.readback_seq,
                        .resource_id = item.resource_id,
                        .resource_version = item.resource_version,
                    });
                    // For UnknownHost, do NOT disarm, and do NOT emit cpu_read_observation
                    ++list_it;
                } else {
                    // Internal readers (WaitRegMemPoll, GpuUploadFromGuestRam, etc.)
                    // Do NOT emit cpu_read_observation, do NOT count as GuestRead
                    ++list_it;
                }
            } else {
                ++list_it;
            }
        }
        if (list.empty()) {
            g_read_watch_interests.erase(it);
        }
    }
    return observed;
}

void RegisterCmdBufferSubmitEnabled(CmdBufferSeq cmd_buf, SubmitSeq submit_seq) noexcept {
    if (cmd_buf == 0) return;
    std::scoped_lock lock{g_cmdbuf_submits_mutex};
    g_cmdbuf_to_submit[cmd_buf] = submit_seq;
}

SubmitSeq LookupCmdBufferSubmitEnabled(CmdBufferSeq cmd_buf) noexcept {
    if (cmd_buf == 0) return 0;
    std::scoped_lock lock{g_cmdbuf_submits_mutex};
    auto it = g_cmdbuf_to_submit.find(cmd_buf);
    return it != g_cmdbuf_to_submit.end() ? it->second : 0;
}

void RegisterPendingReadbackForSubmitEnabled(ReadbackSeq readback_seq, CmdBufferSeq cmd_buf) noexcept {
    if (readback_seq == 0 || cmd_buf == 0) return;
    std::scoped_lock lock{g_cmdbuf_submits_mutex};
    g_cmdbuf_pending_readbacks[cmd_buf].push_back(readback_seq);
}

void PromotePendingReadbacksOnSubmitEnabled(CmdBufferSeq cmd_buf, SubmitSeq submit_seq,
                                           u64 signal_tick) noexcept {
    if (cmd_buf == 0) return;
    std::vector<ReadbackSeq> readbacks;
    {
        std::scoped_lock lock{g_cmdbuf_submits_mutex};
        g_cmdbuf_to_submit[cmd_buf] = submit_seq;
        auto it = g_cmdbuf_pending_readbacks.find(cmd_buf);
        if (it != g_cmdbuf_pending_readbacks.end()) {
            readbacks = std::move(it->second);
            g_cmdbuf_pending_readbacks.erase(it);
        }
    }
    const u64 now = Timestamp();
    for (const auto rb : readbacks) {
        RecordReadbackSubmitEnabled(ReadbackSubmitSample{
            .readback_seq = rb,
            .submit_seq = submit_seq,
            .ready_tick = signal_tick,
            .enqueue_ns = now,
            .copy_bytes = 0,
            .cmd_buffer_seq = cmd_buf,
            .signal_tick = signal_tick,
        });
    }
}

FenceSeq MatchFenceForWaitEnabled(WaitSeq wait_seq, PacketSeq packet_seq, FrameSeq frame_seq,
                                  u32 queue_id, Pm4Engine engine, VAddr wait_addr, u32 ref,
                                  u32 mask, u32 function, PacketSeq prev_pkt, u32 prev_op) noexcept {
    Add(Counter::FenceMatchAttemptCount);
    std::scoped_lock lock{g_active_labels_mutex};

    FenceMatchAttemptSample attempt{
        .wait_seq = wait_seq,
        .packet_seq = packet_seq,
        .frame_seq = frame_seq,
        .queue_id = queue_id,
        .engine = engine,
        .wait_addr = wait_addr,
        .ref = ref,
        .mask = mask,
        .function = function,
        .previous_packet_seq = prev_pkt,
        .previous_opcode = prev_op,
        .candidate_count = 0,
        .nearest_candidate_fence_seq = 0,
        .nearest_candidate_generation = 0,
        .candidate_address_match = false,
        .candidate_value_match = false,
        .candidate_mask_match = false,
        .candidate_function_match = false,
        .packet_distance = 0,
        .result_matched = false,
        .failure_reason = FenceMatchFailure::NoCandidate,
    };

    size_t slot = (wait_addr >> 2) & (ActiveLabelTableCapacity - 1);
    for (size_t probe = 0; probe < 16; ++probe) {
        auto& entry = g_active_labels[slot];
        if (entry.valid && entry.addr == wait_addr) {
            attempt.candidate_count++;
            attempt.nearest_candidate_fence_seq = entry.fence_seq;
            attempt.nearest_candidate_generation = entry.generation;
            attempt.candidate_address_match = true;
            attempt.candidate_mask_match = (entry.mask == mask || mask == 0xffffffff);
            attempt.candidate_value_match = ((entry.value & mask) == (ref & mask));
            if (packet_seq >= entry.create_packet) {
                attempt.packet_distance = static_cast<u32>(packet_seq - entry.create_packet);
            }

            const bool cond = TestWaitCondition(static_cast<u32>(entry.value), function, mask, ref);
            attempt.candidate_function_match = cond;

            if (cond) {
                entry.gpu_wait_count++;
                entry.evidence_bits |= static_cast<u32>(FenceEvidence::MatchedGpuWait);
                attempt.result_matched = true;
                attempt.failure_reason = FenceMatchFailure::None;
                Add(Counter::FenceMatchSuccessCount);
                Add(Counter::WaitMatchedFences);
                RecordFenceMatchAttemptEnabled(attempt);
                return entry.fence_seq;
            } else {
                if (entry.superseded) {
                    attempt.failure_reason = FenceMatchFailure::GenerationSuperseded;
                } else if (!attempt.candidate_mask_match) {
                    attempt.failure_reason = FenceMatchFailure::MaskMismatch;
                } else {
                    attempt.failure_reason = FenceMatchFailure::ReferenceMismatch;
                }
                Add(Counter::FenceMatchFailureCount);
                Add(Counter::WaitUnmatchedFences);
                RecordFenceMatchAttemptEnabled(attempt);

                const auto& dbg = g_fence_candidate_debug[slot];
                u32 map_size = 0;
                bool map_contains = false;
                for (const auto& e : g_active_labels) {
                    if (e.valid) {
                        map_size++;
                        if (e.addr == wait_addr) map_contains = true;
                    }
                }
                RecordFenceMatchDiagnosticEnabled(FenceMatchDiagnosticSample{
                    .wait_seq = wait_seq,
                    .wait_packet_seq = packet_seq,
                    .wait_addr = wait_addr,
                    .wait_ref = ref,
                    .wait_mask = mask,
                    .wait_func = function,
                    .shadow_fence_seq = 0,
                    .last_insert_fence = dbg.last_insert_fence,
                    .last_insert_generation = dbg.last_insert_generation,
                    .last_insert_packet = dbg.last_insert_packet,
                    .last_remove_fence = dbg.last_remove_fence,
                    .last_remove_packet = dbg.last_remove_packet,
                    .last_remove_reason = dbg.last_remove_reason,
                    .last_supersede_fence = dbg.last_superseded_fence,
                    .last_supersede_packet = dbg.last_supersede_packet,
                    .current_candidate_map_contains_address = map_contains,
                    .candidate_map_size = map_size,
                    .packets_since_last_insert = (packet_seq >= dbg.last_insert_packet && dbg.last_insert_packet != 0) ? static_cast<u32>(packet_seq - dbg.last_insert_packet) : 0,
                    .packets_since_last_remove = (packet_seq >= dbg.last_remove_packet && dbg.last_remove_packet != 0) ? static_cast<u32>(packet_seq - dbg.last_remove_packet) : 0,
                });
                return 0;
            }
        }
        if (!entry.valid) {
            break;
        }
        slot = (slot + 1) & (ActiveLabelTableCapacity - 1);
    }

    Add(Counter::FenceMatchFailureCount);
    Add(Counter::WaitUnmatchedFences);
    RecordFenceMatchAttemptEnabled(attempt);

    const auto& dbg = g_fence_candidate_debug[slot];
    u32 map_size = 0;
    bool map_contains = false;
    for (const auto& e : g_active_labels) {
        if (e.valid) {
            map_size++;
            if (e.addr == wait_addr) map_contains = true;
        }
    }
    RecordFenceMatchDiagnosticEnabled(FenceMatchDiagnosticSample{
        .wait_seq = wait_seq,
        .wait_packet_seq = packet_seq,
        .wait_addr = wait_addr,
        .wait_ref = ref,
        .wait_mask = mask,
        .wait_func = function,
        .shadow_fence_seq = 0,
        .last_insert_fence = dbg.last_insert_fence,
        .last_insert_generation = dbg.last_insert_generation,
        .last_insert_packet = dbg.last_insert_packet,
        .last_remove_fence = dbg.last_remove_fence,
        .last_remove_packet = dbg.last_remove_packet,
        .last_remove_reason = dbg.last_remove_reason,
        .last_supersede_fence = dbg.last_superseded_fence,
        .last_supersede_packet = dbg.last_supersede_packet,
        .current_candidate_map_contains_address = map_contains,
        .candidate_map_size = map_size,
        .packets_since_last_insert = (packet_seq >= dbg.last_insert_packet && dbg.last_insert_packet != 0) ? static_cast<u32>(packet_seq - dbg.last_insert_packet) : 0,
        .packets_since_last_remove = (packet_seq >= dbg.last_remove_packet && dbg.last_remove_packet != 0) ? static_cast<u32>(packet_seq - dbg.last_remove_packet) : 0,
    });
    return 0;
}

void ArmConsumerProbeEnabled(FenceSeq fence_seq, WaitSeq wait_seq, PacketSeq wait_pkt,
                             ConsumerProbeKind probe_kind) noexcept {
    if (fence_seq == 0) return;
    std::scoped_lock lock{g_consumer_probe_mutex, g_range_coherence_mutex};
    for (auto& probe : g_consumer_probes) {
        if (!probe.valid) {
            probe.fence_seq = fence_seq;
            probe.wait_seq = wait_seq;
            probe.wait_packet = wait_pkt;
            probe.probe_kind = probe_kind;
            probe.range_count = 0;
            probe.packets_seen = 0;
            probe.producers_seen = 0;
            probe.resolved = false;
            probe.valid = true;

            for (const auto& entry : g_range_coherence) {
                if (entry.valid && probe.range_count < probe.ranges.size()) {
                    probe.ranges[probe.range_count++] = {
                        .addr = entry.addr,
                        .size = entry.size,
                        .resource_id = entry.authoritative_resource,
                        .version = entry.latest_version,
                        .resource_type = entry.authoritative_owner,
                    };
                }
            }
            break;
        }
    }
}

void CheckConsumerOverlapEnabled(ProducerSeq consumer_prod, PacketSeq pkt, ProducerClass consumer_type,
                                 VAddr addr, u64 size, ConsumerAccessPath access_path,
                                 ResourceSeq consumer_res_id, ResourceVersion consumer_res_ver,
                                 ConsumerConfidence confidence,
                                 u64 pipeline_hash, u64 shader_hash) noexcept {
    if (addr == 0 || size == 0) return;
    std::scoped_lock lock{g_consumer_probe_mutex};
    for (auto& probe : g_consumer_probes) {
        if (!probe.valid || probe.resolved) continue;
        for (u32 i = 0; i < probe.range_count; ++i) {
            const auto& r = probe.ranges[i];
            if (addr < r.addr + r.size && addr + size > r.addr) {
                const VAddr overlap_addr = std::max(addr, r.addr);
                const u64 overlap_size = std::min(addr + size, r.addr + r.size) - overlap_addr;
                const u32 dist = pkt >= probe.wait_packet ? static_cast<u32>(pkt - probe.wait_packet) : 0;

                RepresentationTransition trans = RepresentationTransition::SameResource;
                if (r.resource_type == ResourceType::Image && consumer_type == ProducerClass::GraphicsDraw) {
                    trans = RepresentationTransition::ImageToImage;
                } else if (r.resource_type == ResourceType::Image &&
                           (consumer_type == ProducerClass::ComputeDispatch || consumer_type == ProducerClass::StorageBuffer)) {
                    trans = RepresentationTransition::ImageToBuffer;
                } else if (r.resource_type == ResourceType::Buffer && consumer_type == ProducerClass::GraphicsDraw) {
                    trans = RepresentationTransition::BufferToImage;
                }

                if (probe.probe_kind == ConsumerProbeKind::StructuralShadowFence) {
                    Add(Counter::ShadowConsumerProbeHits);
                } else {
                    Add(Counter::MatchedConsumerProbeHits);
                }

                RecordFirstConsumerEnabled(FirstConsumerSample{
                    .fence_seq = probe.fence_seq,
                    .wait_seq = probe.wait_seq,
                    .probe_kind = probe.probe_kind,
                    .producer_seq_source = 0,
                    .producer_seq_consumer = consumer_prod,
                    .consumer_packet_seq = pkt,
                    .packet_distance_from_wait = dist,
                    .source_resource_id = r.resource_id,
                    .source_version = r.version,
                    .consumer_resource_id = consumer_res_id,
                    .consumer_resource_version = consumer_res_ver,
                    .consumer_type = consumer_type,
                    .guest_overlap_addr = overlap_addr,
                    .guest_overlap_size = overlap_size,
                    .access_path = access_path,
                    .transition = trans,
                    .confidence = confidence,
                    .consumer_pipeline_hash = pipeline_hash,
                    .consumer_shader_hash = shader_hash,
                });
                probe.resolved = true;
                probe.valid = false;
                break;
            }
        }
    }
}

void AdvanceConsumerProbesEnabled(PacketSeq current_packet, bool is_present) noexcept {
    std::scoped_lock lock{g_consumer_probe_mutex};
    for (auto& probe : g_consumer_probes) {
        if (!probe.valid) continue;
        probe.packets_seen++;
        if (probe.packets_seen >= 64 || is_present) {
            if (!probe.resolved) {
                RecordFirstConsumerEnabled(FirstConsumerSample{
                    .fence_seq = probe.fence_seq,
                    .wait_seq = probe.wait_seq,
                    .probe_kind = probe.probe_kind,
                    .producer_seq_source = 0,
                    .producer_seq_consumer = 0,
                    .consumer_packet_seq = current_packet,
                    .packet_distance_from_wait = probe.packets_seen,
                    .source_resource_id = probe.range_count > 0 ? probe.ranges[0].resource_id : 0,
                    .source_version = probe.range_count > 0 ? probe.ranges[0].version : 0,
                    .consumer_resource_id = 0,
                    .consumer_resource_version = 0,
                    .consumer_type = ProducerClass::Count,
                    .guest_overlap_addr = probe.range_count > 0 ? probe.ranges[0].addr : 0,
                    .guest_overlap_size = probe.range_count > 0 ? probe.ranges[0].size : 0,
                    .access_path = ConsumerAccessPath::Unknown,
                    .transition = RepresentationTransition::NotObserved,
                    .confidence = ConsumerConfidence::Heuristic,
                    .consumer_pipeline_hash = 0,
                    .consumer_shader_hash = 0,
                });
            }
            probe.valid = false;
        }
    }
}

GuestMemoryWriteOrigin CurrentMemoryWriteOriginEnabled() noexcept {
    return tl_memory.origin;
}

void SetCurrentMemoryWriteOriginEnabled(GuestMemoryWriteOrigin origin, FenceSeq fence_seq) noexcept {
    tl_memory.origin = origin;
    tl_memory.fence_seq = fence_seq;
}

void SetCurrentProducerScopeEnabled(ProducerSeq seq, PacketSeq pkt, ProducerClass type, u32 queue) noexcept {
    tl_producer.seq = seq;
    tl_producer.packet_seq = pkt;
    tl_producer.type = type;
    tl_producer.queue_id = queue;
    tl_producer.start_ns = Timestamp();
    tl_producer.write_range_count = 0;
    tl_producer.write_bytes = 0;
    tl_producer.write_resource_count = 0;
    tl_producer.is_promoted = (type == ProducerClass::ComputeDispatch || type == ProducerClass::ComputeHle);
    tl_producer.active = true;
}

void ClearCurrentProducerScopeEnabled() noexcept {
    if (!tl_producer.active) return;
    const u64 duration = Timestamp() - tl_producer.start_ns;
    RecordProducerRecordEnabled(ProducerRecordSample{
        .producer_seq = tl_producer.seq,
        .packet_seq = tl_producer.packet_seq,
        .frame_seq = CurrentFrameSeqEnabled(),
        .producer_type = tl_producer.type,
        .queue_id = tl_producer.queue_id,
        .stage = 0,
        .shader_hash = 0,
        .pipeline_hash = 0,
        .dispatch_x = 0,
        .dispatch_y = 0,
        .dispatch_z = 0,
        .draw_count = 1,
        .start_ns = tl_producer.start_ns,
        .duration_ns = duration,
        .write_range_count = tl_producer.write_range_count,
        .write_bytes = tl_producer.write_bytes,
        .write_resource_count = tl_producer.write_resource_count,
        .is_promoted = tl_producer.is_promoted,
    });
    tl_producer.active = false;
}

TelemetryProducerScope::TelemetryProducerScope(ProducerSeq seq, PacketSeq pkt, ProducerClass type, u32 queue) noexcept
    : active{true} {
    SetCurrentProducerScopeEnabled(seq, pkt, type, queue);
}

TelemetryProducerScope::~TelemetryProducerScope() {
    if (active) {
        ClearCurrentProducerScopeEnabled();
    }
}

TelemetryMemoryWriteScope::TelemetryMemoryWriteScope(GuestMemoryWriteOrigin origin, FenceSeq fence_seq) noexcept {
    prev_origin = CurrentMemoryWriteOriginEnabled();
    SetCurrentMemoryWriteOriginEnabled(origin, fence_seq);
}

TelemetryMemoryWriteScope::~TelemetryMemoryWriteScope() {
    SetCurrentMemoryWriteOriginEnabled(prev_origin, prev_fence);
}

template <typename Sample, size_t Capacity>
struct RelationalSnapshotItem {
    u64 sequence;
    u64 timestamp_ns;
    Sample sample;
};

template <typename Sample, size_t Capacity>
std::vector<RelationalSnapshotItem<Sample, Capacity>> CollectCommittedRecords(
    const std::unique_ptr<std::array<RelationalRecord<Sample>, Capacity>>& storage,
    const std::atomic<u64>& written_counter) {
    std::vector<RelationalSnapshotItem<Sample, Capacity>> result;
    if (!storage) {
        return result;
    }
    const u64 total_written = written_counter.load(std::memory_order_acquire);
    const u64 start_sequence = total_written > Capacity ? total_written - Capacity : 0;
    result.reserve(static_cast<size_t>(total_written - start_sequence));
    for (u64 sequence = start_sequence; sequence < total_written; ++sequence) {
        const auto& record = (*storage)[sequence & (Capacity - 1)];
        if (record.committed_sequence.load(std::memory_order_acquire) == sequence + 1) {
            result.push_back({
                .sequence = sequence,
                .timestamp_ns = record.timestamp_ns,
                .sample = record.sample,
            });
        }
    }
    return result;
}

struct ZstdStreamWriter {
    std::ofstream file;
    ZSTD_CStream* cstream{nullptr};
    std::vector<char> out_buf;
    bool active{false};

    explicit ZstdStreamWriter(const std::filesystem::path& path, int level = 3) {
        file.open(path, std::ios::binary | std::ios::trunc);
        if (file.is_open()) {
            cstream = ZSTD_createCStream();
            if (cstream) {
                ZSTD_initCStream(cstream, level);
                out_buf.resize(ZSTD_CStreamOutSize());
                active = true;
            }
        }
    }

    ~ZstdStreamWriter() {
        Close();
    }

    void Write(const char* data, size_t size) {
        if (size == 0) return;
        if (!active) {
            if (file.is_open()) {
                file.write(data, size);
            }
            return;
        }
        ZSTD_inBuffer in = {data, size, 0};
        while (in.pos < in.size) {
            ZSTD_outBuffer out = {out_buf.data(), out_buf.size(), 0};
            ZSTD_compressStream(cstream, &out, &in);
            if (out.pos > 0) {
                file.write(out_buf.data(), out.pos);
            }
        }
    }

    void Close() {
        if (active && cstream) {
            ZSTD_inBuffer in = {nullptr, 0, 0};
            size_t rem = 0;
            do {
                ZSTD_outBuffer out = {out_buf.data(), out_buf.size(), 0};
                rem = ZSTD_endStream(cstream, &out);
                if (out.pos > 0) {
                    file.write(out_buf.data(), out.pos);
                }
            } while (rem > 0);
            ZSTD_freeCStream(cstream);
            cstream = nullptr;
            active = false;
        }
        if (file.is_open()) {
            file.close();
        }
    }
};

class TelemetryOutputFormatter {
public:
    explicit TelemetryOutputFormatter(ZstdStreamWriter& writer) : m_writer{writer} {
        m_chunk.reserve(ChunkCapacity);
    }

    ~TelemetryOutputFormatter() {
        Flush();
    }

    template <typename T>
    TelemetryOutputFormatter& operator<<(const T& val) {
        if constexpr (std::is_same_v<T, char>) {
            m_chunk.push_back(val);
        } else if constexpr (std::is_convertible_v<T, std::string_view>) {
            const std::string_view sv{val};
            m_chunk.append(sv);
        } else {
            m_chunk.append(std::to_string(val));
        }
        if (m_chunk.size() >= ChunkFlushThreshold) {
            Flush();
        }
        return *this;
    }

    void Flush() {
        if (!m_chunk.empty()) {
            m_writer.Write(m_chunk.data(), m_chunk.size());
            m_chunk.clear();
        }
    }

private:
    static constexpr size_t ChunkCapacity = 256 * 1024;
    static constexpr size_t ChunkFlushThreshold = 192 * 1024;
    ZstdStreamWriter& m_writer;
    std::string m_chunk;
};
#endif

std::filesystem::path Dump() {
    if (!Enabled() || g_dumping.exchange(true, std::memory_order_acq_rel)) {
        return {};
    }

    std::vector<ThreadRing*> snapshot_rings;
    {
        std::scoped_lock lock{g_rings_mutex};
        snapshot_rings.reserve(g_rings.size());
        for (const auto& ring : g_rings) {
            snapshot_rings.push_back(ring.get());
        }
    }
    if (snapshot_rings.empty()) {
        return {};
    }

    std::array<u64, static_cast<size_t>(Counter::Count)> totals{};
    std::array<u64, 256> opcode_totals{};
    std::array<std::array<u64, HistogramBucketCount>, HistogramCounters.size()>
        histogram_totals{};
    std::vector<EventSnapshot> events;
    events.reserve(snapshot_rings.size() * RingCapacity);
    u64 total_written{};
    u64 total_overwritten{};
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    std::vector<WritebackSnapshot> writeback_records;
    if (g_writeback_records) {
        const u64 writeback_end = g_writeback_sequence.load(std::memory_order_acquire);
        const u64 writeback_begin = writeback_end > WritebackRecordCapacity
                                        ? writeback_end - WritebackRecordCapacity
                                        : 0;
        writeback_records.reserve(static_cast<size_t>(writeback_end - writeback_begin));
        for (u64 sequence = writeback_begin; sequence < writeback_end; ++sequence) {
            const auto& record = (*g_writeback_records)[sequence & (WritebackRecordCapacity - 1)];
            if (record.committed_sequence.load(std::memory_order_acquire) == sequence + 1) {
                writeback_records.push_back({
                    .sequence = sequence,
                    .timestamp_ns = record.timestamp_ns,
                    .sample = record.sample,
                });
            }
        }
    }

    std::vector<FrameRecord> frame_records;
    u64 frame_written{};
    u64 frame_overwritten{};
    if (g_frame_records) {
        std::scoped_lock lock{g_frame_mutex};
        frame_written = g_frame_sequence;
        frame_overwritten = frame_written > FrameRecordCapacity
                                ? frame_written - FrameRecordCapacity
                                : 0;
        frame_records.reserve(static_cast<size_t>(frame_written - frame_overwritten));
        for (u64 sequence = frame_overwritten; sequence < frame_written; ++sequence) {
            frame_records.push_back((*g_frame_records)[sequence & (FrameRecordCapacity - 1)]);
        }
    }

    auto sync_pm4_records = CollectCommittedRecords(g_sync_pm4_records, g_sync_pm4_written);
    auto producer_begin_records = CollectCommittedRecords(g_producer_begin_records, g_producer_begin_written);
    auto producer_end_records = CollectCommittedRecords(g_producer_end_records, g_producer_end_written);
    auto producer_full_records = CollectCommittedRecords(g_producer_full_records, g_producer_full_written);
    auto resource_write_records = CollectCommittedRecords(g_resource_write_records, g_resource_write_written);
    auto fence_create_records = CollectCommittedRecords(g_fence_create_records, g_fence_create_written);
    auto fence_epoch_link_records = CollectCommittedRecords(g_fence_epoch_link_records, g_fence_epoch_link_written);
    auto fence_match_records = CollectCommittedRecords(g_fence_match_records, g_fence_match_written);
    auto fence_match_diagnostic_records = CollectCommittedRecords(g_fence_match_diagnostic_records, g_fence_match_diagnostic_written);
    auto resource_epoch_promoted_records = CollectCommittedRecords(g_resource_epoch_promoted_records, g_resource_epoch_promoted_written);
    auto fence_resource_link_records = CollectCommittedRecords(g_fence_resource_link_records, g_fence_resource_link_written);
    auto wait_create_records = CollectCommittedRecords(g_wait_create_records, g_wait_create_written);
    auto wait_complete_records = CollectCommittedRecords(g_wait_complete_records, g_wait_complete_written);
    auto first_consumer_records = CollectCommittedRecords(g_first_consumer_records, g_first_consumer_written);
    auto fence_classification_records = CollectCommittedRecords(g_fence_classification_records, g_fence_classification_written);
    auto cpu_access_records = CollectCommittedRecords(g_cpu_access_records, g_cpu_access_written);
    auto cpu_label_records = CollectCommittedRecords(g_cpu_label_records, g_cpu_label_written);
    auto cpu_materialization_records = CollectCommittedRecords(g_cpu_materialization_records, g_cpu_materialization_written);
    auto stale_guest_records = CollectCommittedRecords(g_stale_guest_records, g_stale_guest_written);
    auto gpu_alias_records = CollectCommittedRecords(g_gpu_alias_records, g_gpu_alias_written);
    auto readback_schedule_records = CollectCommittedRecords(g_readback_schedule_records, g_readback_schedule_written);
    auto readback_submit_records = CollectCommittedRecords(g_readback_submit_records, g_readback_submit_written);
    auto readback_ready_records = CollectCommittedRecords(g_readback_ready_records, g_readback_ready_written);
    auto readback_commit_records = CollectCommittedRecords(g_readback_commit_records, g_readback_commit_written);

    {
        std::scoped_lock lock{g_source_watches_mutex};
        for (const auto& [key, watch] : g_active_source_watches) {
            RecordReadbackSourceTerminalEnabled(ReadbackSourceTerminalSample{
                .watch_seq = watch.watch_seq,
                .fence_seq = watch.fence_seq,
                .readback_seq = watch.readback_seq,
                .source_resource_id = watch.resource_id,
                .source_resource_version = watch.resource_version,
                .terminal_kind = TerminalKind::SessionEndUnknown,
                .consumer_producer_seq = 0,
                .consumer_packet_seq = 0,
                .consumer_resource_id = 0,
                .consumer_resource_version = 0,
                .access_path = ConsumerAccessPath::Unknown,
                .overlap_addr = watch.guest_addr,
                .overlap_size = watch.size,
                .packets_since_fence = 0,
                .ns_since_fence = 0,
                .correlation_status = CorrelationStatus::Partial,
                .cmd_buffer_seq = 0,
                .submit_seq = 0,
            });
        }
        g_active_source_watches.clear();
    }

    auto readback_source_terminal_records = CollectCommittedRecords(g_readback_source_terminal_records, g_readback_source_terminal_written);
    auto guest_source_consume_records = CollectCommittedRecords(g_guest_source_consume_records, g_guest_source_consume_written);
    auto resource_lineage_records = CollectCommittedRecords(g_resource_lineage_records, g_resource_lineage_written);
    auto cpu_read_observation_records = CollectCommittedRecords(g_cpu_read_observation_records, g_cpu_read_observation_written);
    auto semantic_read_fault_records = CollectCommittedRecords(g_semantic_read_fault_records, g_semantic_read_fault_written);
    auto semantic_read_unknown_records = CollectCommittedRecords(g_semantic_read_unknown_records, g_semantic_read_unknown_written);
    auto semantic_watch_cancel_records = CollectCommittedRecords(g_semantic_watch_cancel_records, g_semantic_watch_cancel_written);
    auto semantic_page_conflict_records = CollectCommittedRecords(g_semantic_page_conflict_records, g_semantic_page_conflict_written);
    auto resource_barrier_link_records = CollectCommittedRecords(g_resource_barrier_link_records, g_resource_barrier_link_written);
    auto acquire_mem_records = CollectCommittedRecords(g_acquire_mem_records, g_acquire_mem_written);
    auto fence_signal_records = CollectCommittedRecords(g_fence_signal_records, g_fence_signal_written);
    auto host_wait_records = CollectCommittedRecords(g_host_wait_records, g_host_wait_written);
    auto submit_records = CollectCommittedRecords(g_submit_records, g_submit_written);
    auto shadow_fence_records = CollectCommittedRecords(g_shadow_fence_records, g_shadow_fence_written);
    auto trace_gap_records = CollectCommittedRecords(g_trace_gap_records, g_trace_gap_written);
    auto ring_health_records = CollectCommittedRecords(g_ring_health_records, g_ring_health_written);
    auto fastpath_candidate_records = CollectCommittedRecords(g_fastpath_candidate_records, g_fastpath_candidate_written);
    auto gpu_authority_create_records = CollectCommittedRecords(g_gpu_authority_create_records, g_gpu_authority_create_written);
    auto virtual_fence_create_records = CollectCommittedRecords(g_virtual_fence_create_records, g_virtual_fence_create_written);
    auto virtual_wait_consume_records = CollectCommittedRecords(g_virtual_wait_consume_records, g_virtual_wait_consume_written);
    auto async_label_signal_records = CollectCommittedRecords(g_async_label_signal_records, g_async_label_signal_written);
    auto authority_gpu_consume_records = CollectCommittedRecords(g_authority_gpu_consume_records, g_authority_gpu_consume_written);
    auto authority_barrier_validation_records = CollectCommittedRecords(g_authority_barrier_validation_records, g_authority_barrier_validation_written);
    auto authority_ram_demand_records = CollectCommittedRecords(g_authority_ram_demand_records, g_authority_ram_demand_written);
    auto lazy_materialize_begin_records = CollectCommittedRecords(g_lazy_materialize_begin_records, g_lazy_materialize_begin_written);
    auto lazy_materialize_end_records = CollectCommittedRecords(g_lazy_materialize_end_records, g_lazy_materialize_end_written);
    auto authority_ram_consume_records = CollectCommittedRecords(g_authority_ram_consume_records, g_authority_ram_consume_written);
    auto authority_cpu_read_records = CollectCommittedRecords(g_authority_cpu_read_records, g_authority_cpu_read_written);
    auto authority_supersede_records = CollectCommittedRecords(g_authority_supersede_records, g_authority_supersede_written);
    auto fastpath_fallback_records = CollectCommittedRecords(g_fastpath_fallback_records, g_fastpath_fallback_written);
    auto conservative_download_decision_records = CollectCommittedRecords(g_conservative_download_decision_records, g_conservative_download_decision_written);
    auto authority_conservative_readback_suppressed_records = CollectCommittedRecords(g_authority_conservative_readback_suppressed_records, g_authority_conservative_readback_suppressed_written);
    auto authority_host_materialize_required_records = CollectCommittedRecords(g_authority_host_materialize_required_records, g_authority_host_materialize_required_written);
    auto fastpath_wait_decision_records = CollectCommittedRecords(g_fastpath_wait_decision_records, g_fastpath_wait_decision_written);
    auto virtual_fence_forced_completion_records = CollectCommittedRecords(g_virtual_fence_forced_completion_records, g_virtual_fence_forced_completion_written);
    auto cpu_to_gpu_label_wait_records = CollectCommittedRecords(g_cpu_to_gpu_label_wait_records, g_cpu_to_gpu_label_wait_written);
#endif

    for (const auto* ring : snapshot_rings) {
        for (size_t i = 0; i < totals.size(); ++i) {
            const auto counter = static_cast<Counter>(i);
            const u64 value = ring->counters[i].load(std::memory_order_relaxed);
            if (IsMaxCounter(counter)) {
                totals[i] = std::max(totals[i], value);
            } else {
                totals[i] += value;
            }
        }
        for (size_t i = 0; i < opcode_totals.size(); ++i) {
            opcode_totals[i] += ring->opcodes[i].load(std::memory_order_relaxed);
        }
        for (size_t histogram = 0; histogram < histogram_totals.size(); ++histogram) {
            for (size_t bucket = 0; bucket < HistogramBucketCount; ++bucket) {
                histogram_totals[histogram][bucket] +=
                    ring->histograms[histogram][bucket].load(std::memory_order_relaxed);
            }
        }

        const u64 end = ring->next_sequence.load(std::memory_order_acquire);
        const u64 begin = end > RingCapacity ? end - RingCapacity : 0;
        total_written += end;
        total_overwritten += begin;
        for (u64 sequence = begin; sequence < end; ++sequence) {
            const auto& slot = ring->events[sequence & RingMask];
            const u64 expected = sequence + 1;
            if (slot.committed_sequence.load(std::memory_order_acquire) != expected) {
                continue;
            }
            EventSnapshot event{
                .timestamp_ns = slot.timestamp_ns.load(std::memory_order_relaxed),
                .arg0 = slot.arg0.load(std::memory_order_relaxed),
                .arg1 = slot.arg1.load(std::memory_order_relaxed),
                .sequence = sequence,
                .thread_id = ring->id,
                .type = static_cast<EventType>(slot.metadata.load(std::memory_order_relaxed) &
                                               0xffff),
            };
            if (slot.committed_sequence.load(std::memory_order_acquire) == expected) {
                events.push_back(event);
            }
        }
    }

    std::ranges::sort(events, {}, &EventSnapshot::timestamp_ns);

    const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                      ("shadps4-telemetry-" + std::to_string(timestamp_ms) + ".csv.zst");

#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    ZstdStreamWriter zwriter{path, 3};
    TelemetryOutputFormatter file{zwriter};
#else
    std::ofstream fallback_file{path, std::ios::binary | std::ios::trunc};
    std::ostream& file = fallback_file;
#endif

    file << "kind,thread,timestamp_ns,name,arg0,arg1,value\n";
    const auto profile = GetCaptureProfile();
    const auto* profile_str = profile == TraceCaptureProfile::SyncSemantic ? "sync_semantic" : "sync_perf";
    file << "metadata,,0,schema_version,0,0,16\n";
    file << "metadata,,0,capture_profile,0,0," << profile_str << "\n";
    file << "metadata,,0,extra_read_faults_enabled,0,0," << (profile == TraceCaptureProfile::SyncSemantic ? "true" : "false") << "\n";
    file << "metadata,,0,session_duration_ns,0,0," << Timestamp() - g_session_start_ns << '\n';
    file << "metadata,,0,ring_capacity,0,0," << RingCapacity << '\n';
    file << "metadata,,0,thread_count,0,0," << snapshot_rings.size() << '\n';
    file << "metadata,,0,event_count,0,0," << events.size() << '\n';
    file << "metadata,,0,event_written,0,0," << total_written << '\n';
    file << "metadata,,0,event_overwritten,0,0," << total_overwritten << '\n';
    file << "metadata,,0,histogram_subdivisions,0,0," << HistogramSubdivisions << '\n';
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    file << "metadata,,0,stage_reason_sample_period,0,0," << StageReasonSamplePeriod << '\n';
    file << "metadata,,0,dynamic_reason_sample_period,0,0," << DynamicReasonSamplePeriod << '\n';
    file << "metadata,,0,descriptor_reason_sample_period,0,0," << DescriptorReasonSamplePeriod << '\n';
    file << "metadata,,0,descriptor_cross_pipeline_sample_period,0,0," << DescriptorCrossPipelineSamplePeriod << '\n';
    file << "metadata,,0,image_find_path_sample_period,0,0," << ImageFindPathSamplePeriod << '\n';
    file << "metadata,,0,staging_detail_sample_period,0,0," << StagingDetailSamplePeriod << '\n';
    file << "metadata,,0,staging_batch_sample_period,0,0," << StagingDetailSamplePeriod << '\n';
    file << "metadata,,0,staging_sparse_copy_sample_period,0,0," << StagingDetailSamplePeriod << '\n';
    file << "metadata,,0,staging_sparse_phase_sample_period,0,0," << TimerSamplePeriod(TimerSite::StagingSparseDenseLookup) << '\n';
    file << "metadata,,0,writeback_record_capacity,0,0," << WritebackRecordCapacity << '\n';
    file << "metadata,,0,writeback_record_count,0,0," << writeback_records.size() << '\n';
    file << "metadata,,0,writeback_record_written,0,0," << g_writeback_sequence.load(std::memory_order_relaxed) << '\n';
    file << "metadata,,0,frame_record_capacity,0,0," << FrameRecordCapacity << '\n';
    file << "metadata,,0,frame_record_count,0,0," << frame_records.size() << '\n';
    file << "metadata,,0,frame_record_written,0,0," << frame_written << '\n';
    file << "metadata,,0,frame_record_overwritten,0,0," << frame_overwritten << '\n';
    file << "metadata,,0,sync_pm4_packet_capacity,0,0," << SyncPm4PacketCapacity << '\n';
    file << "metadata,,0,producer_record_capacity,0,0," << ProducerRecordCapacity << '\n';
    file << "metadata,,0,producer_full_record_capacity,0,0," << ProducerFullRecordCapacity << '\n';
    file << "metadata,,0,resource_write_record_capacity,0,0," << ResourceWriteRecordCapacity << '\n';
    file << "metadata,,0,fence_record_capacity,0,0," << FenceRecordCapacity << '\n';
    file << "metadata,,0,fence_epoch_link_capacity,0,0," << FenceEpochLinkCapacity << '\n';
    file << "metadata,,0,fence_match_capacity,0,0," << FenceMatchAttemptCapacity << '\n';
    file << "metadata,,0,fence_match_diagnostic_capacity,0,0," << FenceMatchDiagnosticCapacity << '\n';
    file << "metadata,,0,resource_epoch_promoted_capacity,0,0," << ResourceEpochPromotedCapacity << '\n';
    file << "metadata,,0,fence_resource_link_capacity,0,0," << FenceResourceLinkCapacity << '\n';
    file << "metadata,,0,wait_record_capacity,0,0," << WaitRecordCapacity << '\n';
    file << "metadata,,0,wait_complete_capacity,0,0," << WaitCompleteCapacity << '\n';
    file << "metadata,,0,first_consumer_capacity,0,0," << FirstConsumerCapacity << '\n';
    file << "metadata,,0,fence_classification_capacity,0,0," << FenceClassificationCapacity << '\n';
    file << "metadata,,0,cpu_access_record_capacity,0,0," << CpuAccessRecordCapacity << '\n';
    file << "metadata,,0,cpu_label_access_capacity,0,0," << CpuLabelAccessCapacity << '\n';
    file << "metadata,,0,cpu_materialization_record_capacity,0,0," << CpuMaterializationRecordCapacity << '\n';
    file << "metadata,,0,stale_guest_attempt_record_capacity,0,0," << StaleGuestAttemptRecordCapacity << '\n';
    file << "metadata,,0,gpu_alias_record_capacity,0,0," << GpuAliasRecordCapacity << '\n';
    file << "metadata,,0,readback_schedule_capacity,0,0," << ReadbackScheduleCapacity << '\n';
    file << "metadata,,0,readback_submit_capacity,0,0," << ReadbackSubmitCapacity << '\n';
    file << "metadata,,0,readback_ready_capacity,0,0," << ReadbackReadyCapacity << '\n';
    file << "metadata,,0,readback_commit_capacity,0,0," << ReadbackCommitCapacity << '\n';
    file << "metadata,,0,readback_source_terminal_capacity,0,0," << ReadbackSourceTerminalCapacity << '\n';
    file << "metadata,,0,guest_source_consume_capacity,0,0," << GuestSourceConsumeCapacity << '\n';
    file << "metadata,,0,resource_lineage_capacity,0,0," << ResourceLineageCapacity << '\n';
    file << "metadata,,0,cpu_read_observation_capacity,0,0," << CpuReadObservationCapacity << '\n';
    file << "metadata,,0,semantic_read_fault_capacity,0,0," << SemanticReadFaultCapacity << '\n';
    file << "metadata,,0,semantic_read_unknown_capacity,0,0," << SemanticReadUnknownCapacity << '\n';
    file << "metadata,,0,resource_barrier_link_capacity,0,0," << ResourceBarrierLinkCapacity << '\n';
    file << "metadata,,0,acquire_mem_capacity,0,0," << AcquireMemCapacity << '\n';
    file << "metadata,,0,fence_signal_capacity,0,0," << FenceSignalCapacity << '\n';
    file << "metadata,,0,host_wait_capacity,0,0," << HostWaitCapacity << '\n';
    file << "metadata,,0,submit_record_capacity,0,0," << SubmitRecordCapacity << '\n';
    file << "metadata,,0,shadow_fence_policy_capacity,0,0," << ShadowFencePolicyCapacity << '\n';
    file << "metadata,,0,semantic_origin_tracking,0,0,true\n";
    file << "metadata,,0,semantic_guest_thread_tracking,0,0,true\n";
    file << "metadata,,0,semantic_guest_rip_tracking,0,0,true\n";
    file << "metadata,,0,semantic_label_watch_arms,0,0," << totals[static_cast<size_t>(Counter::SemanticLabelWatchArms)] << '\n';
    file << "metadata,,0,semantic_data_watch_arms,0,0," << totals[static_cast<size_t>(Counter::SemanticDataWatchArms)] << '\n';
    file << "metadata,,0,semantic_label_watch_pages,0,0," << totals[static_cast<size_t>(Counter::SemanticLabelWatchPages)] << '\n';
    file << "metadata,,0,semantic_data_watch_pages,0,0," << totals[static_cast<size_t>(Counter::SemanticDataWatchPages)] << '\n';
    file << "metadata,,0,semantic_read_faults,0,0," << totals[static_cast<size_t>(Counter::SemanticReadFaults)] << '\n';
    file << "metadata,,0,semantic_read_faults_total,0,0," << totals[static_cast<size_t>(Counter::SemanticReadFaultsTotal)] << '\n';
    file << "metadata,,0,semantic_read_guest_direct,0,0," << totals[static_cast<size_t>(Counter::SemanticReadGuestDirect)] << '\n';
    file << "metadata,,0,semantic_read_guest_hle,0,0," << totals[static_cast<size_t>(Counter::SemanticReadGuestHle)] << '\n';
    file << "metadata,,0,semantic_read_wait_reg_mem_poll,0,0," << totals[static_cast<size_t>(Counter::SemanticReadWaitRegMemPoll)] << '\n';
    file << "metadata,,0,semantic_read_gpu_upload,0,0," << totals[static_cast<size_t>(Counter::SemanticReadGpuUpload)] << '\n';
    file << "metadata,,0,semantic_read_renderer_internal,0,0," << totals[static_cast<size_t>(Counter::SemanticReadRendererInternal)] << '\n';
    file << "metadata,,0,semantic_read_memory_tracker_internal,0,0," << totals[static_cast<size_t>(Counter::SemanticReadMemoryTrackerInternal)] << '\n';
    file << "metadata,,0,semantic_read_telemetry_internal,0,0," << totals[static_cast<size_t>(Counter::SemanticReadTelemetryInternal)] << '\n';
    file << "metadata,,0,semantic_read_unknown_host,0,0," << totals[static_cast<size_t>(Counter::SemanticReadUnknownHost)] << '\n';
    file << "metadata,,0,semantic_label_guest_reads,0,0," << totals[static_cast<size_t>(Counter::SemanticLabelGuestReads)] << '\n';
    file << "metadata,,0,semantic_data_guest_reads,0,0," << totals[static_cast<size_t>(Counter::SemanticDataGuestReads)] << '\n';
    file << "metadata,,0,semantic_precision_unknown_reads,0,0," << totals[static_cast<size_t>(Counter::SemanticPrecisionUnknownReads)] << '\n';
    file << "metadata,,0,cpu_read_observation_count,0,0," << g_cpu_read_observation_written.load(std::memory_order_relaxed) << '\n';
    file << "metadata,,0,semantic_read_fault_count,0,0," << g_semantic_read_fault_written.load(std::memory_order_relaxed) << '\n';
    file << "metadata,,0,semantic_read_unknown_count,0,0," << g_semantic_read_unknown_written.load(std::memory_order_relaxed) << '\n';
    file << "metadata,,0,semantic_watch_cancel_count,0,0," << g_semantic_watch_cancel_written.load(std::memory_order_relaxed) << '\n';
    file << "metadata,,0,semantic_page_conflict_count,0,0," << g_semantic_page_conflict_written.load(std::memory_order_relaxed) << '\n';
    file << "metadata,,0,semantic_page_conflict_writes,0,0," << totals[static_cast<size_t>(Counter::SemanticPageConflictWrites)] << '\n';
    file << "metadata,,0,semantic_watch_cancel_wait_complete,0,0," << totals[static_cast<size_t>(Counter::SemanticWatchCancelWaitComplete)] << '\n';
    file << "metadata,,0,semantic_watch_cancel_overwritten,0,0," << totals[static_cast<size_t>(Counter::SemanticWatchCancelOverwritten)] << '\n';
    file << "metadata,,0,semantic_watch_cancel_page_conflict,0,0," << totals[static_cast<size_t>(Counter::SemanticWatchCancelPageConflict)] << '\n';
    file << "metadata,,0,semantic_fault_livelock_breaks,0,0," << totals[static_cast<size_t>(Counter::SemanticFaultLivelockBreaks)] << '\n';
    file << "metadata,,0,frame_detail_per_present_delta,0,0,1\n";
    for (size_t site = 0; site < TimerSiteCount; ++site) {
        file << "timer_metadata,,0," << TimerNames[site] << ",0,0,"
             << TimerSamplePeriod(static_cast<TimerSite>(site)) << '\n';
    }
    for (size_t stage = 0; stage < TrackedStageCount; ++stage) {
        file << "stage_metadata,,0," << StageNames[stage] << ",0,0," << stage << '\n';
    }
    for (size_t source = 0; source < StagingSourceCount; ++source) {
        file << "staging_source_metadata,,0," << StagingSourceNames[source] << ",0,0,"
             << source << '\n';
    }
    for (size_t kind = 0; kind < StagingMemoryKindCount; ++kind) {
        const auto& detail = g_staging_memory_types[kind];
        if (detail.valid.load(std::memory_order_acquire)) {
            file << "staging_memory_type,,0," << StagingMemoryKindNames[kind] << ','
                 << detail.memory_type.load(std::memory_order_relaxed) << ','
                 << detail.memory_heap.load(std::memory_order_relaxed) << ','
                 << detail.property_flags.load(std::memory_order_relaxed) << '\n';
        }
    }
    const auto write_bit_metadata = [&file](const char* name, size_t bit) {
        file << "bit_metadata,,0," << name << ',' << (u32{1} << bit) << ",0,1\n";
    };
    for (size_t bit = 0; bit < StageReasonBitCount; ++bit) {
        write_bit_metadata(StageReasonBitNames[bit], bit);
    }
    for (size_t bit = 0; bit < DynamicReasonBitCount; ++bit) {
        write_bit_metadata(DynamicReasonBitNames[bit], bit);
    }
    for (size_t bit = 0; bit < DynamicGroupBitCount; ++bit) {
        write_bit_metadata(DynamicGroupBitNames[bit], bit);
    }
    for (size_t bit = 0; bit < DescriptorReasonBitCount; ++bit) {
        write_bit_metadata(DescriptorReasonBitNames[bit], bit);
    }
#endif
    for (const auto* ring : snapshot_rings) {
        const u64 written = ring->next_sequence.load(std::memory_order_relaxed);
        const u64 overwritten = written > RingCapacity ? written - RingCapacity : 0;
        file << "thread," << ring->id << ",0," << CsvSafe(ring->name) << "," << written << ','
             << overwritten << ",0\n";
        for (size_t i = 0; i < totals.size(); ++i) {
            const u64 value = ring->counters[i].load(std::memory_order_relaxed);
            if (value != 0) {
                file << "thread_counter," << ring->id << ",0," << CounterNames[i]
                     << ",0,0," << value << '\n';
            }
        }
        for (size_t histogram = 0; histogram < HistogramCounters.size(); ++histogram) {
            for (size_t bucket = 0; bucket < HistogramBucketCount; ++bucket) {
                const u64 count =
                    ring->histograms[histogram][bucket].load(std::memory_order_relaxed);
                if (count != 0) {
                    const auto [lower, upper] = HistogramBounds(bucket);
                    file << "thread_histogram," << ring->id << ",0,"
                         << HistogramNames[histogram] << ',' << lower << ',' << upper << ','
                         << count << '\n';
                }
            }
        }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
        for (size_t site = 0; site < TimerSiteCount; ++site) {
            const u64 samples = ring->timer_samples[site].load(std::memory_order_relaxed);
            if (samples == 0) {
                continue;
            }
            file << "timer_sample," << ring->id << ",0," << TimerNames[site] << ','
                 << TimerSamplePeriod(static_cast<TimerSite>(site)) << ',' << samples << ','
                 << ring->timer_ns[site].load(std::memory_order_relaxed) << '\n';
            for (size_t bucket = 0; bucket < HistogramBucketCount; ++bucket) {
                const u64 count =
                    ring->timer_histograms[site][bucket].load(std::memory_order_relaxed);
                if (count != 0) {
                    const auto [lower, upper] = HistogramBounds(bucket);
                    file << "timer_histogram," << ring->id << ",0," << TimerNames[site] << ','
                         << lower << ',' << upper << ',' << count << '\n';
                }
            }
        }
        for (size_t stage = 0; stage < TrackedStageCount; ++stage) {
            for (size_t site = 0; site < TimerSiteCount; ++site) {
                const u64 samples =
                    ring->stage_timer_samples[stage][site].load(std::memory_order_relaxed);
                if (samples != 0) {
                    file << "stage_timer_sample," << ring->id << ",0," << TimerNames[site]
                         << ',' << stage << ',' << samples << ','
                         << ring->stage_timer_ns[stage][site].load(std::memory_order_relaxed)
                         << '\n';
                }
            }
            for (size_t reason = 0; reason < StageReasonCount; ++reason) {
                const u64 count =
                    ring->stage_reasons[stage][reason].load(std::memory_order_relaxed);
                if (count != 0) {
                    file << "stage_reason_sample," << ring->id << ",0,uncacheable," << stage
                         << ',' << reason << ',' << count << '\n';
                }
            }
        }
        for (size_t reason = 1; reason < DynamicReasonCount; ++reason) {
            const u64 count = ring->dynamic_reasons[reason].load(std::memory_order_relaxed);
            if (count != 0) {
                file << "dynamic_reason_sample," << ring->id << ",0,miss," << reason << ','
                     << DynamicReasonSamplePeriod << ',' << count << '\n';
            }
        }
        for (size_t groups = 0; groups < DynamicGroupCount; ++groups) {
            const u64 pending =
                ring->dynamic_pending_groups[groups].load(std::memory_order_relaxed);
            if (pending != 0) {
                file << "dynamic_group_sample," << ring->id << ",0,pending," << groups << ','
                     << DynamicReasonSamplePeriod << ',' << pending << '\n';
            }
            const u64 emitted =
                ring->dynamic_emitted_groups[groups].load(std::memory_order_relaxed);
            if (emitted != 0) {
                file << "dynamic_group_sample," << ring->id << ",0,emitted," << groups << ','
                     << DynamicReasonSamplePeriod << ',' << emitted << '\n';
            }
        }
        for (size_t reason = 1; reason < DescriptorReasonCount; ++reason) {
            const u64 count = ring->descriptor_reasons[reason].load(std::memory_order_relaxed);
            if (count != 0) {
                file << "descriptor_reason_sample," << ring->id << ",0,miss," << reason << ','
                     << DescriptorReasonSamplePeriod << ',' << count << '\n';
            }
        }
        for (size_t path_index = 0; path_index < ImageFindPathCount; ++path_index) {
            const u64 count =
                ring->image_find_paths[path_index].load(std::memory_order_relaxed);
            if (count != 0) {
                file << "image_find_path_sample," << ring->id << ",0,"
                     << ImageFindPathNames[path_index] << ',' << ImageFindPathSamplePeriod
                     << ",0," << count << '\n';
            }
        }
        for (size_t site = 0; site < StagingSiteCount; ++site) {
            const auto& detail = ring->staging[site];
            const u64 allocations = detail.allocations.load(std::memory_order_relaxed);
            if (allocations != 0) {
                file << "staging_allocation_sample," << ring->id << ",0,"
                     << StagingSiteNames[site] << ',' << StagingDetailSamplePeriod << ','
                     << allocations << ',' << detail.bytes.load(std::memory_order_relaxed)
                     << '\n';
            }
            for (size_t bucket = 0; bucket < StagingSizeBucketCount; ++bucket) {
                const u64 count = detail.sizes[bucket].load(std::memory_order_relaxed);
                if (count == 0) {
                    continue;
                }
                file << "staging_size_sample," << ring->id << ",0," << StagingSiteNames[site]
                     << ',' << (bucket == 0 ? 0 : (u64{1} << (bucket - 1))) << ','
                     << StagingDetailSamplePeriod << ',' << count << '\n';
            }
            for (size_t source = 0; source < StagingSourceCount; ++source) {
                const u64 records =
                    detail.source_records[source].load(std::memory_order_relaxed);
                if (records != 0) {
                    file << "staging_source_sample," << ring->id << ",0,"
                         << StagingSiteNames[site] << ',' << source << ','
                         << StagingDetailSamplePeriod << ',' << records << '\n';
                    file << "staging_source_bytes," << ring->id << ",0,"
                         << StagingSiteNames[site] << ',' << source << ','
                         << StagingDetailSamplePeriod << ','
                         << detail.source_bytes[source].load(std::memory_order_relaxed) << '\n';
                }
            }
        }
        for (size_t reason = 0; reason < SubmitReasonCount; ++reason) {
            const auto& detail = ring->submits[reason];
            const u64 calls = detail.calls.load(std::memory_order_relaxed);
            if (calls != 0) {
                file << "submit_reason_sample," << ring->id << ",0," << SubmitReasonNames[reason]
                     << ',' << calls << ',' << detail.mutex_wait_ns.load(std::memory_order_relaxed)
                     << ',' << detail.mutex_hold_ns.load(std::memory_order_relaxed) << '\n';
                file << "submit_phase_sample," << ring->id << ",0," << SubmitReasonNames[reason]
                     << ',' << detail.prepare_ns.load(std::memory_order_relaxed) << ','
                     << detail.driver_ns.load(std::memory_order_relaxed) << ','
                     << detail.post_ns.load(std::memory_order_relaxed) << '\n';
            }
        }
        for (size_t trigger = 0; trigger < WritebackTriggerCount; ++trigger) {
            const auto& detail = ring->writeback_drains[trigger];
            const u64 calls = detail[0].load(std::memory_order_relaxed);
            if (calls != 0) {
                file << "writeback_drain_sample," << ring->id << ",0,"
                     << WritebackTriggerNames[trigger] << ',' << calls << ','
                     << detail[1].load(std::memory_order_relaxed) << ','
                     << detail[2].load(std::memory_order_relaxed) << '\n';
            }
        }
        const auto& pm4 = ring->pm4;
        for (size_t engine = 0; engine < Pm4EngineCount; ++engine) {
            for (size_t opcode = 0; opcode < Pm4OpcodeCount; ++opcode) {
                const auto& detail = pm4.opcodes[engine][opcode];
                const u64 packets = detail.packets.load(std::memory_order_relaxed);
                if (packets == 0) {
                    continue;
                }
                file << "pm4_opcode," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                     << engine << ',' << opcode << ',' << packets << '\n';
                file << "pm4_words," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                     << engine << ',' << opcode << ','
                     << detail.words.load(std::memory_order_relaxed) << '\n';
                file << "pm4_predicated," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                     << engine << ',' << opcode << ','
                     << detail.predicated.load(std::memory_order_relaxed) << '\n';
                file << "pm4_shader_compute," << ring->id << ",0," << GetPm4OpcodeName(opcode)
                     << ',' << engine << ',' << opcode << ','
                     << detail.shader_compute.load(std::memory_order_relaxed) << '\n';
                file << "pm4_depth_max," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                     << engine << ',' << opcode << ','
                     << detail.max_depth.load(std::memory_order_relaxed) << '\n';
                for (size_t word_count = 0; word_count < Pm4WordBucketCount; ++word_count) {
                    const u64 count = detail.word_counts[word_count].load(std::memory_order_relaxed);
                    if (count != 0) {
                        file << "pm4_word_bucket," << ring->id << ",0,"
                             << GetPm4OpcodeName(opcode) << ',' << engine << ',' << word_count
                             << ',' << count << '\n';
                    }
                }
            }
            for (size_t space = 0; space < Pm4RegisterSpaceCount; ++space) {
                for (size_t reg = 0; reg < Pm4RegisterCount; ++reg) {
                    const auto& detail = pm4.registers[engine][space][reg];
                    const u64 packets = detail.packets.load(std::memory_order_relaxed);
                    if (packets == 0) {
                        continue;
                    }
                    file << "pm4_register," << ring->id << ",0," << GetPm4RegisterName(space, reg)
                         << ',' << engine << ',' << space << ',' << packets << '\n';
                    file << "pm4_register_words," << ring->id << ",0,"
                         << GetPm4RegisterName(space, reg) << ',' << engine << ',' << space << ','
                         << detail.words.load(std::memory_order_relaxed) << '\n';
                    file << "pm4_register_changed," << ring->id << ",0,"
                         << GetPm4RegisterName(space, reg) << ',' << engine << ',' << space << ','
                         << detail.changed.load(std::memory_order_relaxed) << '\n';
                }
            }
        }
        for (const auto& control : pm4.controls) {
            const u64 hash = control.hash.load(std::memory_order_relaxed);
            if (hash == 0) {
                continue;
            }
            const u64 packets = control.packets.load(std::memory_order_relaxed);
            if (packets == 0) {
                continue;
            }
            const auto [engine, queue_id, opcode, depth] = UnpackPm4Identity(control.identity);
            file << "pm4_control," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                 << engine << ',' << queue_id << ',' << packets << '\n';
            file << "pm4_control_detail," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                 << depth << ',' << control.control0 << ',' << control.control1 << '\n';
        }
        for (const auto& wait : pm4.waits) {
            const u64 hash = wait.hash.load(std::memory_order_relaxed);
            if (hash == 0) {
                continue;
            }
            const u64 packets = wait.packets.load(std::memory_order_relaxed);
            if (packets == 0) {
                continue;
            }
            const auto [engine, queue_id, opcode, depth] = UnpackPm4Identity(wait.identity);
            file << "pm4_wait," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                 << engine << ',' << queue_id << ',' << packets << '\n';
            file << "pm4_wait_location," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                 << depth << ',' << wait.location << ',' << wait.control << '\n';
            file << "pm4_wait_condition," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                 << wait.reference << ',' << wait.mask << ',' << wait.poll_interval << '\n';
            file << "pm4_wait_results," << ring->id << ",0," << GetPm4OpcodeName(opcode) << ','
                 << wait.failed_tests.load(std::memory_order_relaxed) << ','
                 << wait.immediate_passes.load(std::memory_order_relaxed) << ','
                 << wait.vo_sleeps.load(std::memory_order_relaxed) << '\n';
        }
        const u64 register_overflow = pm4.register_overflow.load(std::memory_order_relaxed);
        if (register_overflow != 0) {
            file << "pm4_register_overflow," << ring->id << ",0,overflow,0,0," << register_overflow
                 << '\n';
        }
        const u64 control_overflow = pm4.control_overflow.load(std::memory_order_relaxed);
        if (control_overflow != 0) {
            file << "pm4_control_overflow," << ring->id << ",0,overflow,0,0," << control_overflow
                 << '\n';
        }
        const u64 wait_overflow = pm4.wait_overflow.load(std::memory_order_relaxed);
        if (wait_overflow != 0) {
            file << "pm4_wait_overflow," << ring->id << ",0,overflow,0,0," << wait_overflow
                 << '\n';
        }
#endif
    }
    for (size_t histogram = 0; histogram < HistogramCounters.size(); ++histogram) {
        for (size_t bucket = 0; bucket < HistogramBucketCount; ++bucket) {
            const u64 count = histogram_totals[histogram][bucket];
            if (count != 0) {
                const auto [lower, upper] = HistogramBounds(bucket);
                file << "histogram,,0," << HistogramNames[histogram] << ',' << lower << ','
                     << upper << ',' << count << '\n';
            }
        }
    }
    for (const auto& event : events) {
        const size_t type = static_cast<size_t>(event.type);
        const auto name = type < EventNames.size() ? EventNames[type] : "unknown";
        file << "sync_event," << event.thread_id << ',' << event.timestamp_ns << ',' << name << ','
             << event.arg0 << ',' << event.arg1 << ',' << event.sequence << '\n';
    }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    for (const auto& record : frame_records) {
        file << "frame," << record.sequence << ',' << record.timestamp_ns << ",timing,"
             << record.frame_id << ',' << record.interval_ns << ',' << record.present_duration_ns
             << '\n';
        for (size_t index = 0; index < FrameCounters.size(); ++index) {
            file << "frame_counter," << record.sequence << ',' << record.timestamp_ns << ','
                 << CounterNames[static_cast<size_t>(FrameCounters[index])] << ','
                 << record.frame_id << ',' << record.interval_ns << ',' << record.counters[index]
                 << '\n';
        }
        for (size_t site = 0; site < TimerSiteCount; ++site) {
            if (record.timer_samples[site] != 0) {
                file << "frame_timer," << record.sequence << ',' << record.timestamp_ns << ','
                     << TimerNames[site] << ',' << record.timer_samples[site] << ','
                     << TimerSamplePeriod(static_cast<TimerSite>(site)) << ','
                     << record.timer_ns[site] << '\n';
            }
        }
        for (size_t stage = 0; stage < TrackedStageCount; ++stage) {
            for (size_t site = 0; site < StageFrameTimerCount; ++site) {
                const u64 samples = record.details.stage_timer_samples[stage][site];
                if (samples != 0) {
                    file << "frame_stage_timer," << record.sequence << ','
                         << record.timestamp_ns << ',' << TimerNames[site] << ',' << stage << ','
                         << samples << ',' << record.details.stage_timer_ns[stage][site] << '\n';
                }
            }
            for (size_t bit = 0; bit < StageReasonBitCount; ++bit) {
                const u64 count = record.details.stage_reason_bits[stage][bit];
                if (count != 0) {
                    file << "frame_stage_reason," << record.sequence << ','
                         << record.timestamp_ns << ",uncacheable," << stage << ','
                         << (u32{1} << bit) << ',' << count << '\n';
                }
            }
        }
        for (size_t bit = 0; bit < DynamicReasonBitCount; ++bit) {
            const u64 count = record.details.dynamic_reason_bits[bit];
            if (count != 0) {
                file << "frame_dynamic_reason," << record.sequence << ',' << record.timestamp_ns
                     << ",miss," << (u32{1} << bit) << ',' << DynamicReasonSamplePeriod << ','
                     << count << '\n';
            }
        }
        for (size_t bit = 0; bit < DynamicGroupBitCount; ++bit) {
            const u32 group = u32{1} << bit;
            const u64 pending = record.details.dynamic_pending_group_bits[bit];
            if (pending != 0) {
                file << "frame_dynamic_group," << record.sequence << ',' << record.timestamp_ns
                     << ",pending," << group << ',' << DynamicReasonSamplePeriod << ',' << pending
                     << '\n';
            }
            const u64 emitted = record.details.dynamic_emitted_group_bits[bit];
            if (emitted != 0) {
                file << "frame_dynamic_group," << record.sequence << ',' << record.timestamp_ns
                     << ",emitted," << group << ',' << DynamicReasonSamplePeriod << ',' << emitted
                     << '\n';
            }
        }
        for (size_t bit = 0; bit < DescriptorReasonBitCount; ++bit) {
            const u64 count = record.details.descriptor_reason_bits[bit];
            if (count != 0) {
                file << "frame_descriptor_reason," << record.sequence << ','
                     << record.timestamp_ns << ",miss," << (u32{1} << bit) << ','
                     << DescriptorReasonSamplePeriod << ',' << count << '\n';
            }
        }
        for (size_t path = 0; path < ImageFindPathCount; ++path) {
            const u64 count = record.details.image_find_paths[path];
            if (count != 0) {
                file << "frame_image_find_path," << record.sequence << ','
                     << record.timestamp_ns << ',' << ImageFindPathNames[path] << ','
                     << ImageFindPathSamplePeriod << ",0," << count << '\n';
            }
        }
        for (size_t site = 0; site < StagingSiteCount; ++site) {
            const u64 allocations = record.details.staging_allocations[site];
            if (allocations != 0) {
                file << "frame_staging_sample," << record.sequence << ','
                     << record.timestamp_ns << ',' << StagingSiteNames[site] << ',' << allocations
                     << ',' << StagingDetailSamplePeriod << ','
                     << record.details.staging_bytes[site] << '\n';
            }
            for (size_t source = 0; source < StagingSourceCount; ++source) {
                const u64 records = record.details.staging_source_records[site][source];
                if (records != 0) {
                    file << "frame_staging_source," << record.sequence << ','
                         << record.timestamp_ns << ',' << StagingSiteNames[site] << ',' << source
                         << ',' << records << ','
                         << record.details.staging_source_bytes[site][source] << '\n';
                }
            }
        }
        for (size_t reason = 0; reason < SubmitReasonCount; ++reason) {
            const auto& submit = record.details.submits[reason];
            if (submit.calls != 0) {
                file << "frame_submit_reason," << record.sequence << ',' << record.timestamp_ns
                     << ',' << SubmitReasonNames[reason] << ',' << submit.calls << ','
                     << submit.mutex_wait_ns << ',' << submit.mutex_hold_ns << '\n';
                file << "frame_submit_phase," << record.sequence << ',' << record.timestamp_ns
                     << ',' << SubmitReasonNames[reason] << ',' << submit.prepare_ns << ','
                     << submit.driver_ns << ',' << submit.post_ns << '\n';
            }
        }
    }
    for (const auto& record : writeback_records) {
        const auto& sample = record.sample;
        const size_t trigger = static_cast<size_t>(sample.trigger);
        const size_t writer = static_cast<size_t>(sample.writer);
        const auto trigger_name = trigger < WritebackTriggerNames.size()
                                      ? WritebackTriggerNames[trigger]
                                      : "unknown";
        const auto writer_name = writer < ImageWriterNames.size() ? ImageWriterNames[writer]
                                                                   : "unknown";
        file << "writeback_image," << record.sequence << ',' << record.timestamp_ns << ','
             << trigger_name << ',' << sample.image_index << ',' << sample.image_uid << ','
             << sample.content_epoch << '\n';
        file << "writeback_writer," << record.sequence << ',' << record.timestamp_ns << ','
             << writer_name << ',' << sample.flags << ',' << sample.previous_epoch << ','
             << sample.previous_backing << '\n';
        file << "writeback_range," << record.sequence << ',' << record.timestamp_ns
             << ",guest_range," << sample.guest_address << ',' << sample.download_bytes << ','
             << sample.backing_image << '\n';
        file << "writeback_trigger," << record.sequence << ',' << record.timestamp_ns
             << ",control," << sample.trigger_control << ',' << sample.trigger_data_control << ','
             << sample.queued_images << '\n';
    }
    for (const auto& record : sync_pm4_records) {
        file << "sync_pm4_packet," << record.sample.packet_seq << ',' << record.timestamp_ns << ','
             << GetPm4OpcodeName(record.sample.opcode) << ',' << record.sample.frame_seq << ','
             << record.sample.cmd_buffer_seq << ',' << record.sample.queue_id << ','
             << static_cast<u32>(record.sample.engine) << ',' << record.sample.ib_depth << ','
             << record.sample.opcode << ',' << record.sample.packet_addr << ','
             << record.sample.label_addr << ',' << record.sample.label_value << ','
             << record.sample.event_type << ',' << record.sample.event_index << ','
             << record.sample.interrupt_select << ',' << record.sample.data_select << ','
             << record.sample.cache_action << '\n';
    }
    for (const auto& record : producer_begin_records) {
        const size_t type_idx = static_cast<size_t>(record.sample.producer_type);
        const auto type_name = type_idx < ProducerClassNames.size() ? ProducerClassNames[type_idx] : "unknown";
        file << "producer_begin," << record.sample.producer_seq << ',' << record.timestamp_ns << ','
             << type_name << ',' << record.sample.packet_seq << ',' << record.sample.frame_seq << ','
             << record.sample.queue_id << ',' << record.sample.stage << ',' << record.sample.shader_hash << ','
             << record.sample.pipeline_hash << ',' << record.sample.dispatch_x << ',' << record.sample.dispatch_y << ','
             << record.sample.dispatch_z << ',' << record.sample.draw_count << '\n';
    }
    for (const auto& record : producer_end_records) {
        file << "producer_end," << record.sample.producer_seq << ',' << record.timestamp_ns << ",end,"
             << record.sample.write_range_count << ',' << record.sample.write_bytes << ','
             << record.sample.write_resource_count << '\n';
    }
    for (const auto& record : producer_full_records) {
        const size_t type_idx = static_cast<size_t>(record.sample.producer_type);
        const auto type_name = type_idx < ProducerClassNames.size() ? ProducerClassNames[type_idx] : "unknown";
        file << "producer_record," << record.sample.producer_seq << ',' << record.timestamp_ns << ','
             << type_name << ',' << record.sample.packet_seq << ',' << record.sample.frame_seq << ','
             << record.sample.queue_id << ',' << record.sample.stage << ',' << record.sample.shader_hash << ','
             << record.sample.pipeline_hash << ',' << record.sample.dispatch_x << ',' << record.sample.dispatch_y << ','
             << record.sample.dispatch_z << ',' << record.sample.draw_count << ',' << record.sample.duration_ns << ','
             << record.sample.write_range_count << ',' << record.sample.write_bytes << ','
             << record.sample.write_resource_count << ',' << (record.sample.is_promoted ? 1 : 0) << '\n';
    }
    for (const auto& record : resource_write_records) {
        const size_t kind_idx = static_cast<size_t>(record.sample.write_kind);
        const auto kind_name = kind_idx < ResourceWriteKindNames.size() ? ResourceWriteKindNames[kind_idx] : "unknown";
        file << "resource_write," << record.sample.producer_seq << ',' << record.timestamp_ns << ','
             << kind_name << ',' << record.sample.resource_id << ',' << record.sample.version << ','
             << record.sample.guest_addr << ',' << record.sample.guest_size << ',' << record.sample.vk_handle_id << ','
             << record.sample.format << ',' << record.sample.width << ',' << record.sample.height << ','
             << record.sample.depth << ',' << record.sample.pitch << ',' << record.sample.tiling << ','
             << record.sample.stage << ',' << record.sample.queue_id << '\n';
    }
    for (const auto& record : fence_create_records) {
        const size_t kind_idx = static_cast<size_t>(record.sample.fence_kind);
        const auto kind_name = kind_idx < FenceKindNames.size() ? FenceKindNames[kind_idx] : "unknown";
        const size_t scope_idx = static_cast<size_t>(record.sample.stage_scope);
        const auto scope_name = scope_idx < FenceStageScopeNames.size() ? FenceStageScopeNames[scope_idx] : "unknown";
        const size_t class_idx = static_cast<size_t>(record.sample.classification_initial);
        const auto class_name = class_idx < FenceClassificationNames.size() ? FenceClassificationNames[class_idx] : "unknown";
        const size_t status_idx = static_cast<size_t>(record.sample.correlation_status);
        const auto status_name = status_idx < CorrelationStatusNames.size() ? CorrelationStatusNames[status_idx] : "unknown";
        file << "fence_create," << record.sample.fence_seq << ',' << record.timestamp_ns << ','
             << kind_name << ',' << record.sample.packet_seq << ',' << record.sample.frame_seq << ','
             << record.sample.label_addr << ',' << record.sample.label_value << ',' << record.sample.label_size << ','
             << record.sample.queue_id << ',' << scope_name << ',' << record.sample.interrupt_select << ','
             << record.sample.interrupt_id << ',' << record.sample.data_select << ','
              << record.sample.producer_begin_seq << ',' << record.sample.producer_end_seq << ','
              << record.sample.candidate_write_epoch_count << ',' << record.sample.candidate_write_bytes << ','
              << class_name << ',' << record.sample.classification_reason_bits << ',' << status_name << ','
              << record.sample.generation << ',' << record.sample.raw_event_type << ','
              << record.sample.decoded_event_type << ',' << record.sample.command << '\n';
    }
    for (const auto& record : fence_epoch_link_records) {
        const size_t reason_idx = static_cast<size_t>(record.sample.reason);
        const auto reason_name = reason_idx < FenceLinkReasonNames.size() ? FenceLinkReasonNames[reason_idx] : "unknown";
        const size_t rk_idx = static_cast<size_t>(record.sample.resource_kind);
        const auto rk_name = rk_idx < ResourceTypeNames.size() ? ResourceTypeNames[rk_idx] : "unknown";
        const size_t wk_idx = static_cast<size_t>(record.sample.write_kind);
        const auto wk_name = wk_idx < ResourceWriteKindNames.size() ? ResourceWriteKindNames[wk_idx] : "unknown";
        file << "fence_epoch_link," << record.sample.fence_seq << ',' << record.timestamp_ns << ','
             << reason_name << ',' << record.sample.producer_seq << ',' << record.sample.resource_id << ','
             << record.sample.version << ',' << record.sample.guest_addr << ',' << record.sample.guest_size << ','
             << rk_name << ',' << wk_name << ',' << record.sample.queue_id << ',' << record.sample.stage << '\n';
    }
    for (const auto& record : fence_match_records) {
        const size_t fail_idx = static_cast<size_t>(record.sample.failure_reason);
        const auto fail_name = fail_idx < FenceMatchFailureNames.size() ? FenceMatchFailureNames[fail_idx] : "unknown";
        file << "fence_match_attempt," << record.sample.wait_seq << ',' << record.timestamp_ns << ','
             << (record.sample.result_matched ? "matched" : "failed") << ','
             << record.sample.packet_seq << ',' << record.sample.frame_seq << ',' << record.sample.queue_id << ','
             << static_cast<u32>(record.sample.engine) << ',' << record.sample.wait_addr << ','
             << record.sample.ref << ',' << record.sample.mask << ',' << record.sample.function << ','
             << record.sample.previous_packet_seq << ',' << record.sample.previous_opcode << ','
             << record.sample.candidate_count << ',' << record.sample.nearest_candidate_fence_seq << ','
             << record.sample.nearest_candidate_generation << ',' << (record.sample.candidate_address_match ? 1 : 0) << ','
             << (record.sample.candidate_value_match ? 1 : 0) << ',' << (record.sample.candidate_mask_match ? 1 : 0) << ','
             << (record.sample.candidate_function_match ? 1 : 0) << ',' << record.sample.packet_distance << ','
             << fail_name << '\n';
    }
    for (const auto& record : fence_match_diagnostic_records) {
        const size_t rem_idx = static_cast<size_t>(record.sample.last_remove_reason);
        const auto rem_name = rem_idx < CandidateRemoveReasonNames.size() ? CandidateRemoveReasonNames[rem_idx] : "unknown";
        file << "fence_match_diagnostic," << record.sample.wait_seq << ',' << record.timestamp_ns << ','
             << record.sample.wait_packet_seq << ',' << record.sample.wait_addr << ','
             << record.sample.wait_ref << ',' << record.sample.wait_mask << ',' << record.sample.wait_func << ','
             << record.sample.shadow_fence_seq << ',' << record.sample.last_insert_fence << ','
             << record.sample.last_insert_generation << ',' << record.sample.last_insert_packet << ','
             << record.sample.last_remove_fence << ',' << record.sample.last_remove_packet << ','
             << rem_name << ',' << record.sample.last_supersede_fence << ','
             << record.sample.last_supersede_packet << ',' << (record.sample.current_candidate_map_contains_address ? 1 : 0) << ','
             << record.sample.candidate_map_size << ',' << record.sample.packets_since_last_insert << ','
             << record.sample.packets_since_last_remove << '\n';
    }
    for (const auto& record : resource_epoch_promoted_records) {
        const size_t prom_idx = static_cast<size_t>(record.sample.promotion_reason);
        const auto prom_name = prom_idx < PromotionReasonNames.size() ? PromotionReasonNames[prom_idx] : "unknown";
        const size_t kind_idx = static_cast<size_t>(record.sample.write_kind);
        const auto kind_name = kind_idx < ResourceWriteKindNames.size() ? ResourceWriteKindNames[kind_idx] : "unknown";
        file << "resource_epoch_promoted," << record.sample.resource_id << ',' << record.timestamp_ns << ','
             << prom_name << ',' << record.sample.resource_version << ',' << record.sample.producer_seq << ','
             << record.sample.producer_packet_seq << ',' << record.sample.producer_timestamp << ','
             << record.sample.guest_addr << ',' << record.sample.size << ',' << kind_name << '\n';
    }
    for (const auto& record : fence_resource_link_records) {
        const size_t rsn_idx = static_cast<size_t>(record.sample.reason);
        const auto rsn_name = rsn_idx < FenceLinkReasonNames.size() ? FenceLinkReasonNames[rsn_idx] : "unknown";
        const size_t rk_idx = static_cast<size_t>(record.sample.resource_kind);
        const auto rk_name = rk_idx < ResourceTypeNames.size() ? ResourceTypeNames[rk_idx] : "unknown";
        const size_t wk_idx = static_cast<size_t>(record.sample.write_kind);
        const auto wk_name = wk_idx < ResourceWriteKindNames.size() ? ResourceWriteKindNames[wk_idx] : "unknown";
        const size_t conf_idx = static_cast<size_t>(record.sample.confidence);
        const auto conf_name = conf_idx < ConsumerConfidenceNames.size() ? ConsumerConfidenceNames[conf_idx] : "unknown";
        file << "fence_resource_link," << record.sample.fence_seq << ',' << record.timestamp_ns << ','
             << rsn_name << ',' << record.sample.resource_id << ',' << record.sample.resource_version << ','
             << record.sample.producer_seq << ',' << record.sample.guest_addr << ',' << record.sample.size << ','
             << rk_name << ',' << wk_name << ',' << conf_name << '\n';
    }
    for (const auto& record : wait_create_records) {
        const size_t conf_idx = static_cast<size_t>(record.sample.matched_fence_confidence);
        const auto conf_name = conf_idx < WaitConfidenceNames.size() ? WaitConfidenceNames[conf_idx] : "unknown";
        const size_t basis_idx = static_cast<size_t>(record.sample.shadow_basis);
        const auto basis_name = basis_idx < ShadowBasisNames.size() ? ShadowBasisNames[basis_idx] : "unknown";
        const size_t status_idx = static_cast<size_t>(record.sample.matcher_status);
        const auto status_name = status_idx < CorrelationStatusNames.size() ? CorrelationStatusNames[status_idx] : "unknown";
        const size_t fail_idx = static_cast<size_t>(record.sample.failure_reason);
        const auto fail_name = fail_idx < FenceMatchFailureNames.size() ? FenceMatchFailureNames[fail_idx] : "unknown";
        file << "wait_create," << record.sample.wait_seq << ',' << record.timestamp_ns << ",wait,"
             << record.sample.packet_seq << ',' << record.sample.frame_seq << ',' << record.sample.queue_id << ','
             << static_cast<u32>(record.sample.engine) << ',' << record.sample.wait_addr << ','
             << record.sample.ref << ',' << record.sample.mask << ',' << record.sample.function << ','
             << record.sample.matched_fence_seq << ',' << record.sample.matched_generation << ','
             << conf_name << ',' << record.sample.commands_since_fence << ','
             << record.sample.packets_since_fence << ',' << record.sample.producer_ops_between << ','
             << record.sample.shadow_fence_seq << ',' << basis_name << ',' << status_name << ','
             << fail_name << '\n';
    }
    for (const auto& record : wait_complete_records) {
        file << "wait_complete," << record.sample.wait_seq << ',' << record.timestamp_ns << ",complete,"
             << record.sample.fence_seq << ',' << record.sample.start_timestamp << ','
             << record.sample.end_timestamp << ',' << record.sample.duration_ns << ','
             << record.sample.spin_iterations << ',' << record.sample.yield_count << ','
             << record.sample.exit_reason << ',' << record.sample.value_at_begin << ','
             << record.sample.value_at_end << ',' << record.sample.gpu_completed_tick_begin << ','
             << record.sample.gpu_completed_tick_end << ',' << record.sample.shadow_fence_seq << '\n';
    }
    for (const auto& record : first_consumer_records) {
        const size_t trans_idx = static_cast<size_t>(record.sample.transition);
        const auto trans_name = trans_idx < RepresentationTransitionNames.size() ? RepresentationTransitionNames[trans_idx] : "unknown";
        const size_t type_idx = static_cast<size_t>(record.sample.consumer_type);
        const auto type_name = type_idx < ProducerClassNames.size() ? ProducerClassNames[type_idx] : "unknown";
        const size_t probe_idx = static_cast<size_t>(record.sample.probe_kind);
        const auto probe_name = probe_idx < ConsumerProbeKindNames.size() ? ConsumerProbeKindNames[probe_idx] : "unknown";
        const size_t path_idx = static_cast<size_t>(record.sample.access_path);
        const auto path_name = path_idx < ConsumerAccessPathNames.size() ? ConsumerAccessPathNames[path_idx] : "unknown";
        const size_t conf_idx = static_cast<size_t>(record.sample.confidence);
        const auto conf_name = conf_idx < ConsumerConfidenceNames.size() ? ConsumerConfidenceNames[conf_idx] : "unknown";
        file << "first_consumer," << record.sample.fence_seq << ',' << record.timestamp_ns << ','
             << trans_name << ',' << record.sample.wait_seq << ','
             << record.sample.producer_seq_source << ',' << record.sample.producer_seq_consumer << ','
             << record.sample.consumer_packet_seq << ',' << record.sample.packet_distance_from_wait << ','
             << record.sample.source_resource_id << ',' << record.sample.source_version << ','
             << record.sample.consumer_resource_id << ',' << type_name << ','
             << record.sample.guest_overlap_addr << ',' << record.sample.guest_overlap_size << ','
             << path_name << ',' << probe_name << ',' << record.sample.consumer_resource_version << ','
             << conf_name << ',' << record.sample.consumer_pipeline_hash << ','
             << record.sample.consumer_shader_hash << '\n';
    }
    for (const auto& record : fence_classification_records) {
        const size_t class_idx = static_cast<size_t>(record.sample.classification_final);
        const auto class_name = class_idx < FenceClassificationNames.size() ? FenceClassificationNames[class_idx] : "unknown";
        const size_t conf_idx = static_cast<size_t>(record.sample.confidence);
        const auto conf_name = conf_idx < WaitConfidenceNames.size() ? WaitConfidenceNames[conf_idx] : "unknown";
        file << "fence_classification," << record.sample.fence_seq << ',' << record.timestamp_ns << ','
             << class_name << ',' << record.sample.gpu_wait_count << ',' << record.sample.cpu_label_read_count << ','
             << record.sample.cpu_label_write_count << ',' << (record.sample.irq_requested ? 1 : 0) << ','
             << record.sample.cpu_protected_data_read_count << ',' << conf_name << ',' << record.sample.reason_bits << '\n';
    }
    for (const auto& record : cpu_access_records) {
        const size_t access_idx = static_cast<size_t>(record.sample.access_type);
        const auto access_name = access_idx < CpuAccessTypeNames.size() ? CpuAccessTypeNames[access_idx] : "unknown";
        const size_t res_idx = static_cast<size_t>(record.sample.authoritative_owner);
        const auto res_name = res_idx < ResourceTypeNames.size() ? ResourceTypeNames[res_idx] : "unknown";
        const size_t origin_idx = static_cast<size_t>(record.sample.write_origin);
        const auto origin_name = origin_idx < GuestMemoryWriteOriginNames.size() ? GuestMemoryWriteOriginNames[origin_idx] : "unknown";
        file << "cpu_memory_access," << record.sample.cpu_access_seq << ',' << record.timestamp_ns << ','
             << access_name << ',' << record.sample.frame_seq << ',' << record.sample.thread_id << ','
             << record.sample.guest_addr << ',' << record.sample.size << ',' << (record.sample.is_label_range ? 1 : 0) << ','
             << record.sample.matched_fence_seq << ',' << record.sample.latest_version << ',' << record.sample.host_version << ','
             << res_name << ',' << record.sample.authoritative_resource << ',' << record.sample.authoritative_tick << ','
             << origin_name << '\n';
    }
    for (const auto& record : cpu_label_records) {
        const size_t access_idx = static_cast<size_t>(record.sample.access_type);
        const auto access_name = access_idx < CpuAccessTypeNames.size() ? CpuAccessTypeNames[access_idx] : "unknown";
        file << "cpu_label_access," << record.sample.fence_seq << ',' << record.timestamp_ns << ','
             << access_name << ',' << record.sample.generation << ','
             << record.sample.guest_addr << ',' << record.sample.guest_thread_id << '\n';
    }
    for (const auto& record : cpu_materialization_records) {
        const size_t res_idx = static_cast<size_t>(record.sample.source_resource_type);
        const auto res_name = res_idx < ResourceTypeNames.size() ? ResourceTypeNames[res_idx] : "unknown";
        file << "cpu_read_materialization," << record.sample.cpu_access_seq << ',' << record.timestamp_ns << ",materialization,"
             << record.sample.guest_addr << ',' << record.sample.size << ',' << record.sample.latest_version << ','
             << record.sample.host_version << ',' << record.sample.producer_seq << ',' << record.sample.producer_tick << ','
             << record.sample.source_resource_id << ',' << res_name << ','
             << (record.sample.readback_already_scheduled ? 1 : 0) << ','
             << (record.sample.readback_ready ? 1 : 0) << '\n';
    }
    for (const auto& record : stale_guest_records) {
        const size_t act_idx = static_cast<size_t>(record.sample.fallback_action);
        const auto act_name = act_idx < FallbackActionNames.size() ? FallbackActionNames[act_idx] : "unknown";
        const size_t cons_idx = static_cast<size_t>(record.sample.consumer_type);
        const auto cons_name = cons_idx < ResourceTypeNames.size() ? ResourceTypeNames[cons_idx] : "unknown";
        const size_t auth_idx = static_cast<size_t>(record.sample.authoritative_resource_type);
        const auto auth_name = auth_idx < ResourceTypeNames.size() ? ResourceTypeNames[auth_idx] : "unknown";
        file << "stale_guest_attempt," << record.sample.frame_seq << ',' << record.timestamp_ns << ','
             << act_name << ',' << record.sample.packet_seq << ',' << record.sample.guest_addr << ','
             << record.sample.size << ',' << record.sample.consumer_resource_id << ',' << cons_name << ','
             << record.sample.host_version << ',' << record.sample.latest_version << ','
             << record.sample.authoritative_resource_id << ',' << auth_name << '\n';
    }
    for (const auto& record : gpu_alias_records) {
        const size_t copy_idx = static_cast<size_t>(record.sample.copy_kind);
        const auto copy_name = copy_idx < AliasCopyKindNames.size() ? AliasCopyKindNames[copy_idx] : "unknown";
        file << "gpu_alias_materialize," << record.sample.submit_seq << ',' << record.timestamp_ns << ','
             << copy_name << ',' << record.sample.source_resource_id << ',' << record.sample.source_version << ','
             << record.sample.dest_resource_id << ',' << record.sample.dest_previous_version << ','
             << record.sample.dest_new_version << ',' << record.sample.guest_addr << ','
             << record.sample.size << ',' << record.sample.packet_seq << '\n';
    }
    for (const auto& record : readback_schedule_records) {
        const size_t rsn_idx = static_cast<size_t>(record.sample.reason);
        const auto rsn_name = rsn_idx < ReadbackReasonNames.size() ? ReadbackReasonNames[rsn_idx] : "unknown";
        const size_t status_idx = static_cast<size_t>(record.sample.correlation_status);
        const auto status_name = status_idx < CorrelationStatusNames.size() ? CorrelationStatusNames[status_idx] : "unknown";
        file << "readback_schedule," << record.sample.readback_seq << ',' << record.timestamp_ns << ','
             << rsn_name << ',' << record.sample.fence_seq << ',' << record.sample.resource_id << ','
             << record.sample.version << ',' << record.sample.guest_addr << ',' << record.sample.size << ','
             << record.sample.download_offset << ',' << record.sample.producer_tick << ','
             << record.sample.schedule_tick << ',' << status_name << '\n';
    }
    for (const auto& record : readback_submit_records) {
        file << "readback_submit," << record.sample.readback_seq << ',' << record.timestamp_ns << ",submit,"
             << record.sample.submit_seq << ',' << record.sample.ready_tick << ','
             << record.sample.enqueue_ns << ',' << record.sample.copy_bytes << ','
             << record.sample.cmd_buffer_seq << ',' << record.sample.signal_tick << '\n';
    }
    for (const auto& record : readback_ready_records) {
        file << "readback_ready," << record.sample.readback_seq << ',' << record.timestamp_ns << ",ready,"
             << record.sample.ready_tick << ',' << record.sample.schedule_to_ready_ns << ','
             << record.sample.submit_to_ready_ns << ',' << record.sample.submit_seq << '\n';
    }
    for (const auto& record : readback_commit_records) {
        file << "readback_commit," << record.sample.readback_seq << ',' << record.timestamp_ns << ",commit,"
             << record.sample.memcpy_ns << ',' << record.sample.invalidate_ns << ','
             << record.sample.guest_addr << ',' << record.sample.size << ',' << record.sample.version << ','
             << record.sample.host_version_before << ',' << record.sample.host_version_after << ','
             << record.sample.fence_seq << ',' << record.sample.resource_id << '\n';
    }
    for (const auto& record : readback_source_terminal_records) {
        const size_t term_idx = static_cast<size_t>(record.sample.terminal_kind);
        const auto term_name = term_idx < TerminalKindNames.size() ? TerminalKindNames[term_idx] : "unknown";
        const size_t path_idx = static_cast<size_t>(record.sample.access_path);
        const auto path_name = path_idx < ConsumerAccessPathNames.size() ? ConsumerAccessPathNames[path_idx] : "unknown";
        const size_t status_idx = static_cast<size_t>(record.sample.correlation_status);
        const auto status_name = status_idx < CorrelationStatusNames.size() ? CorrelationStatusNames[status_idx] : "unknown";
        file << "readback_source_terminal," << record.sample.watch_seq << ',' << record.timestamp_ns << ','
             << term_name << ',' << record.sample.fence_seq << ',' << record.sample.readback_seq << ','
             << record.sample.source_resource_id << ',' << record.sample.source_resource_version << ','
             << record.sample.consumer_producer_seq << ',' << record.sample.consumer_packet_seq << ','
             << record.sample.consumer_resource_id << ',' << record.sample.consumer_resource_version << ','
             << path_name << ',' << record.sample.overlap_addr << ',' << record.sample.overlap_size << ','
             << record.sample.packets_since_fence << ',' << record.sample.ns_since_fence << ','
             << status_name << ',' << record.sample.cmd_buffer_seq << ',' << record.sample.submit_seq << '\n';
    }
    for (const auto& record : guest_source_consume_records) {
        const size_t path_idx = static_cast<size_t>(record.sample.path);
        const auto path_name = path_idx < GuestSourceConsumePathNames.size() ? GuestSourceConsumePathNames[path_idx] : "unknown";
        const size_t kind_idx = static_cast<size_t>(record.sample.destination_kind);
        const auto kind_name = kind_idx < ResourceTypeNames.size() ? ResourceTypeNames[kind_idx] : "unknown";
        file << "guest_source_consume," << record.sample.origin_readback_seq << ',' << record.timestamp_ns << ','
             << path_name << ',' << record.sample.origin_resource_id << ',' << record.sample.origin_resource_version << ','
             << record.sample.guest_addr << ',' << record.sample.size << ','
             << record.sample.consumer_producer_seq << ',' << record.sample.consumer_packet_seq << ','
             << kind_name << ',' << record.sample.destination_resource_id << '\n';
    }
    for (const auto& record : resource_lineage_records) {
        const size_t kind_idx = static_cast<size_t>(record.sample.kind);
        const auto kind_name = kind_idx < ResourceLineageKindNames.size() ? ResourceLineageKindNames[kind_idx] : "unknown";
        file << "resource_lineage," << record.sample.source_resource_id << ',' << record.timestamp_ns << ','
             << kind_name << ',' << record.sample.source_resource_version << ','
             << record.sample.destination_resource_id << ',' << record.sample.destination_resource_version << '\n';
    }
    for (const auto& record : cpu_read_observation_records) {
        const size_t kind_idx = static_cast<size_t>(record.sample.watch_kind);
        const auto kind_name = kind_idx < ReadWatchKindNames.size() ? ReadWatchKindNames[kind_idx] : "unknown";
        file << "cpu_read_observation," << record.sample.thread_id << ',' << record.timestamp_ns << ','
             << kind_name << ',' << record.sample.fence_seq << ',' << record.sample.generation << ','
             << record.sample.readback_seq << ',' << record.sample.resource_id << ','
             << record.sample.resource_version << ',' << record.sample.fault_addr << ','
             << record.sample.guest_range_addr << ',' << record.sample.guest_range_size << ','
             << record.sample.ns_since_fence << ',' << record.sample.packets_since_fence << '\n';
    }
    for (const auto& record : semantic_read_fault_records) {
        const size_t origin_idx = static_cast<size_t>(record.sample.origin);
        const auto origin_name = origin_idx < SemanticReadOriginNames.size() ? SemanticReadOriginNames[origin_idx] : "unknown";
        const size_t kind_idx = static_cast<size_t>(record.sample.watch_kind);
        const auto kind_name = kind_idx < ReadWatchKindNames.size() ? ReadWatchKindNames[kind_idx] : "unknown";
        const size_t hv_origin_idx = static_cast<size_t>(record.sample.host_version_origin);
        const auto hv_origin_name = hv_origin_idx < HostVersionOriginNames.size() ? HostVersionOriginNames[hv_origin_idx] : "unknown";
        file << "semantic_read_fault," << record.sample.thread_id << ',' << record.timestamp_ns << ','
             << origin_name << ',' << record.sample.rip << ',' << record.sample.fault_addr << ','
             << record.sample.access_size << ',' << kind_name << ',' << record.sample.fence_seq << ','
             << record.sample.readback_seq << ',' << record.sample.resource_id << ','
             << record.sample.resource_version << ',' << record.sample.host_version << ','
             << hv_origin_name << '\n';
    }
    for (const auto& record : semantic_read_unknown_records) {
        const size_t kind_idx = static_cast<size_t>(record.sample.watch_kind);
        const auto kind_name = kind_idx < ReadWatchKindNames.size() ? ReadWatchKindNames[kind_idx] : "unknown";
        file << "semantic_read_unknown," << record.sample.thread_id << ',' << record.timestamp_ns << ",unknown,"
             << record.sample.rip << ',' << record.sample.fault_addr << ',' << record.sample.access_size << ','
             << kind_name << ',' << record.sample.fence_seq << ',' << record.sample.readback_seq << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << '\n';
    }
    for (const auto& record : semantic_watch_cancel_records) {
        const size_t kind_idx = static_cast<size_t>(record.sample.watch_kind);
        const auto kind_name = kind_idx < ReadWatchKindNames.size() ? ReadWatchKindNames[kind_idx] : "unknown";
        const size_t reason_idx = static_cast<size_t>(record.sample.reason);
        const auto reason_name = reason_idx < SemanticWatchCancelReasonNames.size() ? SemanticWatchCancelReasonNames[reason_idx] : "unknown";
        file << "semantic_watch_cancel," << record.sample.fence_seq << ',' << record.timestamp_ns << ','
             << reason_name << ',' << record.sample.page << ',' << record.sample.guest_addr << ','
             << record.sample.size << ',' << kind_name << ',' << record.sample.readback_seq << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << ','
             << record.sample.remaining_page_owners << '\n';
    }
    for (const auto& record : semantic_page_conflict_records) {
        const size_t kind_idx = static_cast<size_t>(record.sample.watch_kind);
        const auto kind_name = kind_idx < ReadWatchKindNames.size() ? ReadWatchKindNames[kind_idx] : "unknown";
        file << "semantic_page_conflict," << record.sample.thread_id << ',' << record.timestamp_ns << ','
             << record.sample.rip << ',' << record.sample.fault_addr << ',' << record.sample.write_size << ','
             << record.sample.watched_addr << ',' << record.sample.watched_size << ',' << kind_name << ','
             << record.sample.fence_seq << ',' << record.sample.readback_seq << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << '\n';
    }
    for (const auto& record : resource_barrier_link_records) {
        file << "resource_barrier_link," << record.sample.resource_id << ',' << record.timestamp_ns << ','
             << record.sample.reason_path << ',' << record.sample.resource_version << ','
             << record.sample.fence_seq << ',' << record.sample.readback_seq << ','
             << record.sample.cmd_buffer_seq << ',' << record.sample.submit_seq << ','
             << record.sample.old_layout << ',' << record.sample.new_layout << ','
             << record.sample.src_stage << ',' << record.sample.src_access << ','
             << record.sample.dst_stage << ',' << record.sample.dst_access << ','
             << record.sample.subresource_or_range << '\n';
    }
    for (const auto& record : acquire_mem_records) {
        const size_t eng_idx = static_cast<size_t>(record.sample.engine);
        const auto eng_name = GetPm4EngineName(eng_idx);
        file << "acquire_mem," << record.sample.packet_seq << ',' << record.timestamp_ns << ','
             << eng_name << ',' << record.sample.frame_seq << ',' << record.sample.queue_id << ','
             << record.sample.cp_coher_cntl << ',' << record.sample.cp_coher_size_lo << ','
             << record.sample.cp_coher_size_hi << ',' << record.sample.cp_coher_base_lo << ','
             << record.sample.cp_coher_base_hi << ',' << record.sample.poll_interval << ','
             << record.sample.decoded_base_addr << ',' << record.sample.decoded_size << ','
             << record.sample.decoded_cache_flags << ',' << record.sample.previous_wait_seq << ','
             << record.sample.shadow_fence_seq << '\n';
    }
    for (const auto& record : fence_signal_records) {
        const size_t origin_idx = static_cast<size_t>(record.sample.signal_origin);
        const auto origin_name = origin_idx < MemoryWriteOriginNames.size() ? MemoryWriteOriginNames[origin_idx] : "unknown";
        file << "fence_signal," << record.sample.fence_seq << ',' << record.timestamp_ns << ",signal,"
             << record.sample.ready_tick << ',' << record.sample.signal_ns << ','
             << record.sample.writeback_count << ',' << record.sample.writeback_bytes << ','
             << record.sample.gpu_ready_to_signal_ns << ',' << record.sample.irq << ','
             << record.sample.generation << ',' << record.sample.label_addr << ','
             << record.sample.label_value << ',' << origin_name << '\n';
    }
    for (const auto& record : host_wait_records) {
        const size_t rsn_idx = static_cast<size_t>(record.sample.reason);
        const auto rsn_name = rsn_idx < HostWaitReasonNames.size() ? HostWaitReasonNames[rsn_idx] : "unknown";
        file << "host_wait," << record.sample.wait_seq << ',' << record.timestamp_ns << ','
             << rsn_name << ',' << record.sample.requested_tick << ',' << record.sample.current_gpu_tick_before << ','
             << record.sample.cpu_tick << ',' << record.sample.queue_depth_estimate << ','
             << record.sample.start_ns << ',' << record.sample.duration_ns << ','
             << record.sample.fence_seq << ',' << record.sample.readback_seq << ','
             << record.sample.packet_seq << ',' << record.sample.submit_seq << '\n';
    }
    for (const auto& record : submit_records) {
        const size_t rsn_idx = static_cast<size_t>(record.sample.reason);
        const auto rsn_name = rsn_idx < SubmitReasonNames.size() ? SubmitReasonNames[rsn_idx] : "unknown";
        file << "submit_record," << record.sample.submit_seq << ',' << record.timestamp_ns << ','
             << rsn_name << ',' << record.sample.frame_seq << ',' << record.sample.signal_tick << ','
             << record.sample.command_count << ',' << record.sample.producer_count << ','
             << record.sample.copy_count << ',' << record.sample.copy_bytes << ','
             << record.sample.pending_fence_count << ',' << record.sample.pending_readback_count << ','
             << record.sample.cpu_ahead_ticks << ',' << record.sample.gpu_completed_tick << ','
             << record.sample.scheduler_id << ',' << record.sample.queue_role << ','
             << record.sample.cmd_buffer_seq << '\n';
    }
    for (const auto& record : shadow_fence_records) {
        const size_t class_idx = static_cast<size_t>(record.sample.would_classify);
        const auto class_name = class_idx < FenceClassificationNames.size() ? FenceClassificationNames[class_idx] : "unknown";
        const size_t conf_idx = static_cast<size_t>(record.sample.confidence);
        const auto conf_name = conf_idx < WaitConfidenceNames.size() ? WaitConfidenceNames[conf_idx] : "unknown";
        file << "shadow_fence_policy," << record.sample.fence_seq << ",0,shadow,"
             << class_name << ',' << (record.sample.would_skip_host_readback ? 1 : 0) << ','
             << record.sample.reason_bits << ',' << conf_name << '\n';
    }
    for (const auto& record : trace_gap_records) {
        const size_t rsn_idx = static_cast<size_t>(record.sample.reason);
        const auto rsn_name = rsn_idx < TraceGapReasonNames.size() ? TraceGapReasonNames[rsn_idx] : "unknown";
        file << "trace_gap," << record.sample.stream_id << ",0,gap,"
             << record.sample.first_missing_seq << ',' << record.sample.last_missing_seq << ','
             << rsn_name << ',' << record.sample.count << '\n';
    }
    for (const auto& record : ring_health_records) {
        file << "ring_health," << record.sample.stream_id << ",0,health,"
             << record.sample.capacity_records << ',' << record.sample.capacity_bytes << ','
             << record.sample.seen << ',' << record.sample.emitted << ','
             << record.sample.filtered << ',' << record.sample.sampled << ','
             << record.sample.written << ',' << record.sample.overwritten << ','
             << record.sample.dropped_writer_backpressure << ','
             << record.sample.first_event_seq << ',' << record.sample.last_event_seq << ','
             << record.sample.first_timestamp << ',' << record.sample.last_timestamp << ','
             << record.sample.oldest_retained_timestamp << '\n';
    }
    for (const auto& record : fastpath_candidate_records) {
        const size_t el_idx = static_cast<size_t>(record.sample.eligibility);
        const auto el_name = el_idx < FastpathEligibilityNames.size() ? FastpathEligibilityNames[el_idx] : "unknown";
        const size_t rej_idx = static_cast<size_t>(record.sample.reject_reason);
        const auto rej_name = rej_idx < FastpathRejectReasonNames.size() ? FastpathRejectReasonNames[rej_idx] : "unknown";
        file << "fastpath_candidate," << record.sample.candidate_seq << ',' << record.timestamp_ns << ','
             << el_name << ',' << rej_name << ',' << record.sample.producer_seq << ','
             << record.sample.producer_packet_seq << ',' << record.sample.producer_kind << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << ','
             << record.sample.image_id << ',' << record.sample.image_uid << ','
             << record.sample.guest_addr << ',' << record.sample.size << ','
             << record.sample.fence_seq << ',' << record.sample.eos_packet_seq << ','
             << record.sample.label_addr << ',' << record.sample.label_value << ','
             << record.sample.label_num_bytes << ',' << record.sample.wait_packet_seq << ','
             << record.sample.wait_compare << ',' << record.sample.wait_ref << ','
             << record.sample.wait_mask << ',' << record.sample.acquire_packet_seq << ','
             << record.sample.acquire_raw_cntl << '\n';
    }
    for (const auto& record : gpu_authority_create_records) {
        file << "gpu_authority_create," << record.sample.authority_seq << ',' << record.timestamp_ns << ','
             << record.sample.candidate_seq << ',' << record.sample.resource_id << ','
             << record.sample.resource_version << ',' << record.sample.image_id << ','
             << record.sample.image_uid << ',' << record.sample.guest_begin << ','
             << record.sample.guest_end << ',' << record.sample.size << ','
             << record.sample.producer_seq << ',' << record.sample.producer_packet_seq << ','
             << record.sample.producer_tick << ',' << record.sample.cmd_buffer_seq << ','
             << record.sample.submit_seq << ',' << record.sample.fence_seq << ','
             << record.sample.virtual_fence_seq << ',' << record.sample.label_addr << ','
             << record.sample.label_generation << '\n';
    }
    for (const auto& record : virtual_fence_create_records) {
        file << "virtual_fence_create," << record.sample.virtual_fence_seq << ',' << record.timestamp_ns << ','
             << record.sample.authority_seq << ',' << record.sample.fence_seq << ','
             << record.sample.label_addr << ',' << record.sample.label_generation << ','
             << record.sample.expected_value << ',' << record.sample.producer_tick << ','
             << record.sample.producer_packet_seq << ',' << record.sample.eos_packet_seq << ','
             << record.sample.wait_packet_seq << ',' << record.sample.acquire_packet_seq << '\n';
    }
    for (const auto& record : virtual_wait_consume_records) {
        const size_t res_idx = static_cast<size_t>(record.sample.result);
        const auto res_name = res_idx < VirtualWaitResultNames.size() ? VirtualWaitResultNames[res_idx] : "unknown";
        file << "virtual_wait_consume," << record.sample.virtual_fence_seq << ',' << record.timestamp_ns << ','
             << res_name << ',' << record.sample.authority_seq << ',' << record.sample.wait_seq << ','
             << record.sample.wait_packet_seq << ',' << record.sample.label_addr << ','
             << record.sample.label_generation << ',' << record.sample.producer_tick << ','
             << static_cast<u32>(record.sample.producer_tick_complete_at_consume) << '\n';
    }
    for (const auto& record : async_label_signal_records) {
        const size_t act_idx = static_cast<size_t>(record.sample.action);
        const auto act_name = act_idx < AsyncLabelActionNames.size() ? AsyncLabelActionNames[act_idx] : "unknown";
        file << "async_label_signal," << record.sample.virtual_fence_seq << ',' << record.timestamp_ns << ','
             << act_name << ',' << record.sample.authority_seq << ',' << record.sample.label_addr << ','
             << record.sample.scheduled_generation << ',' << record.sample.current_generation << ','
             << record.sample.producer_tick << ',' << record.sample.current_completed_tick << '\n';
    }
    for (const auto& record : authority_gpu_consume_records) {
        file << "authority_gpu_consume," << record.sample.authority_seq << ',' << record.timestamp_ns << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << ','
             << record.sample.image_id << ',' << record.sample.image_uid << ','
             << record.sample.consumer_seq << ',' << record.sample.consumer_packet_seq << ','
             << record.sample.consumer_kind << ',' << record.sample.requested_access << ','
             << record.sample.requested_layout << ',' << record.sample.producer_tick << ','
             << record.sample.consumer_cmd_buffer_seq << ',' << record.sample.consumer_submit_seq << '\n';
    }
    for (const auto& record : authority_barrier_validation_records) {
        file << "authority_barrier_validation," << record.sample.authority_seq << ',' << record.timestamp_ns << ','
             << record.sample.consumer_seq << ',' << record.sample.resource_id << ','
             << record.sample.resource_version << ',' << record.sample.old_layout << ','
             << record.sample.new_layout << ',' << record.sample.src_stage << ','
             << record.sample.src_access << ',' << record.sample.dst_stage << ','
             << record.sample.dst_access << ',' << record.sample.subresource_range << ','
             << static_cast<u32>(record.sample.valid_write_dependency) << '\n';
    }
    for (const auto& record : authority_ram_demand_records) {
        const size_t path_idx = static_cast<size_t>(record.sample.path);
        const auto path_name = path_idx < GuestSourceConsumePathNames.size() ? GuestSourceConsumePathNames[path_idx] : "unknown";
        const size_t dst_idx = static_cast<size_t>(record.sample.destination_kind);
        const auto dst_name = dst_idx < ResourceTypeNames.size() ? ResourceTypeNames[dst_idx] : "unknown";
        file << "authority_ram_demand," << record.sample.ram_demand_seq << ',' << record.timestamp_ns << ','
             << path_name << ',' << record.sample.ram_demand_group_seq << ','
             << record.sample.authority_seq << ',' << record.sample.resource_id << ','
             << record.sample.resource_version << ',' << record.sample.authority_begin << ','
             << record.sample.authority_end << ',' << record.sample.request_addr << ','
             << record.sample.request_size << ',' << record.sample.overlap_begin << ','
             << record.sample.overlap_size << ',' << record.sample.consumer_seq << ','
             << record.sample.consumer_producer_seq << ',' << record.sample.consumer_packet_seq << ','
             << dst_name << ',' << record.sample.destination_resource_id << ','
             << static_cast<u32>(record.sample.authority_state_before) << '\n';
    }
    for (const auto& record : lazy_materialize_begin_records) {
        file << "lazy_materialize_begin," << record.sample.materialize_seq << ',' << record.timestamp_ns << ','
             << record.sample.ram_demand_seq << ',' << record.sample.authority_seq << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << ','
             << record.sample.image_id << ',' << record.sample.image_uid << ','
             << record.sample.producer_tick << ',' << record.sample.current_completed_tick << ','
             << record.sample.guest_begin << ',' << record.sample.guest_end << ','
             << record.sample.reason << '\n';
    }
    for (const auto& record : lazy_materialize_end_records) {
        const size_t res_idx = static_cast<size_t>(record.sample.result);
        const auto res_name = res_idx < LazyMaterializeResultNames.size() ? LazyMaterializeResultNames[res_idx] : "unknown";
        file << "lazy_materialize_end," << record.sample.materialize_seq << ',' << record.timestamp_ns << ','
             << res_name << ',' << record.sample.ram_demand_seq << ','
             << record.sample.authority_seq << ',' << record.sample.producer_tick << ','
             << record.sample.completed_tick << ',' << record.sample.bytes_materialized << ','
             << record.sample.guest_begin << ',' << record.sample.guest_end << ','
             << static_cast<u32>(record.sample.host_current_after) << ','
             << static_cast<s32>(record.sample.validation_bytes_equal) << '\n';
    }
    for (const auto& record : authority_ram_consume_records) {
        const size_t path_idx = static_cast<size_t>(record.sample.path);
        const auto path_name = path_idx < GuestSourceConsumePathNames.size() ? GuestSourceConsumePathNames[path_idx] : "unknown";
        file << "authority_ram_consume," << record.sample.ram_demand_seq << ',' << record.timestamp_ns << ','
             << path_name << ',' << record.sample.authority_seq << ','
             << record.sample.materialize_seq << ',' << record.sample.consumer_seq << ','
             << record.sample.request_addr << ',' << record.sample.request_size << ','
             << record.sample.overlap_begin << ',' << record.sample.overlap_size << ','
             << static_cast<u32>(record.sample.host_current) << ','
             << static_cast<u32>(record.sample.producer_tick_complete) << ','
             << static_cast<u32>(record.sample.materialize_success) << '\n';
    }
    for (const auto& record : authority_cpu_read_records) {
        const size_t orig_idx = static_cast<size_t>(record.sample.origin);
        const auto orig_name = orig_idx < SemanticReadOriginNames.size() ? SemanticReadOriginNames[orig_idx] : "unknown";
        file << "authority_cpu_read," << record.sample.authority_seq << ',' << record.timestamp_ns << ','
             << orig_name << ',' << record.sample.fault_addr << ','
             << record.sample.guest_read_begin << ',' << record.sample.guest_read_size << ','
             << record.sample.materialize_seq << ','
             << static_cast<u32>(record.sample.resumed_after_materialize) << '\n';
    }
    for (const auto& record : authority_supersede_records) {
        file << "authority_supersede," << record.sample.old_authority_seq << ',' << record.timestamp_ns << ','
             << record.sample.new_authority_seq << ',' << record.sample.overlap_begin << ','
             << record.sample.overlap_size << ',' << record.sample.old_resource_id << ','
             << record.sample.old_resource_version << ',' << record.sample.new_resource_id << ','
             << record.sample.new_resource_version << ',' << static_cast<u32>(record.sample.old_host_current) << '\n';
    }
    for (const auto& record : fastpath_fallback_records) {
        const size_t ph_idx = static_cast<size_t>(record.sample.phase);
        const auto ph_name = ph_idx < FastpathFallbackPhaseNames.size() ? FastpathFallbackPhaseNames[ph_idx] : "unknown";
        const size_t rsn_idx = static_cast<size_t>(record.sample.reason);
        const auto rsn_name = rsn_idx < FastpathFallbackReasonNames.size() ? FastpathFallbackReasonNames[rsn_idx] : "unknown";
        file << "fastpath_fallback," << record.sample.candidate_seq << ',' << record.timestamp_ns << ','
             << ph_name << ',' << rsn_name << ',' << record.sample.authority_seq_if_created << '\n';
    }
    for (const auto& record : conservative_download_decision_records) {
        const size_t dec_idx = static_cast<size_t>(record.sample.decision);
        const auto dec_name = dec_idx < ConservativeDownloadDecisionNames.size() ? ConservativeDownloadDecisionNames[dec_idx] : "unknown";
        file << "conservative_download_decision," << record.timestamp_ns << ','
             << record.sample.trigger << ',' << record.sample.fence_seq << ','
             << record.sample.image_id << ',' << record.sample.image_uid << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << ','
             << record.sample.guest_addr << ',' << record.sample.size << ','
             << record.sample.authority_seq << ',' << static_cast<u32>(record.sample.authority_state) << ','
             << dec_name << ',' << record.sample.reason << ','
             << static_cast<u32>(record.sample.readback_schedule_seen) << ','
             << record.sample.readback_seq << '\n';
    }
    for (const auto& record : authority_conservative_readback_suppressed_records) {
        file << "authority_conservative_readback_suppressed," << record.sample.authority_seq << ','
             << record.sample.resource_id << ',' << record.sample.resource_version << ','
             << record.sample.trigger << ',' << record.sample.fence_seq << ','
             << record.sample.guest_addr << ',' << record.sample.size << ','
             << record.sample.image_id << ',' << record.sample.image_uid << '\n';
    }
    for (const auto& record : authority_host_materialize_required_records) {
        file << "authority_host_materialize_required," << record.sample.authority_seq << ','
             << record.sample.reason << ',' << record.sample.request_addr << ','
             << record.sample.request_size << ',' << record.sample.overlap_addr << ','
             << record.sample.overlap_size << '\n';
    }
    for (const auto& record : fastpath_wait_decision_records) {
        const size_t dec_idx = static_cast<size_t>(record.sample.decision);
        const auto dec_name = dec_idx < FastpathWaitDecisionNames.size() ? FastpathWaitDecisionNames[dec_idx] : "unknown";
        file << "fastpath_wait_decision," << record.sample.candidate_seq << ','
             << record.sample.virtual_fence_seq << ',' << record.sample.authority_seq << ','
             << record.sample.wait_seq << ',' << record.sample.wait_packet_seq << ','
             << record.sample.label_addr << ',' << record.sample.label_generation << ','
             << record.sample.ref << ',' << record.sample.mask << ','
             << record.sample.compare << ',' << record.sample.producer_tick << ','
             << record.sample.current_tick << ',' << dec_name << ','
             << record.sample.reason << '\n';
    }
    for (const auto& record : virtual_fence_forced_completion_records) {
        const size_t rsn_idx = static_cast<size_t>(record.sample.reason);
        const auto rsn_name = rsn_idx < VirtualFenceForcedCompletionReasonNames.size() ? VirtualFenceForcedCompletionReasonNames[rsn_idx] : "unknown";
        file << "virtual_fence_forced_completion," << record.sample.virtual_fence_seq << ','
             << record.sample.authority_seq << ',' << rsn_name << ','
             << record.sample.producer_tick << ',' << static_cast<u32>(record.sample.was_submitted) << ','
             << static_cast<u32>(record.sample.waited) << ',' << record.sample.duration_ns << '\n';
    }
    for (const auto& record : cpu_to_gpu_label_wait_records) {
        file << "cpu_to_gpu_label_wait," << record.sample.wait_seq << ','
             << record.sample.frame_id << ',' << record.sample.label_addr << ','
             << record.sample.ref << ',' << record.sample.mask << ','
             << record.sample.wait_begin << ',' << record.sample.wait_end << ','
             << record.sample.duration_ns << ',' << record.sample.last_guest_write_ts << ','
             << record.sample.last_guest_write_value << ',' << record.sample.writer_thread_id << ','
             << record.sample.delta_write_to_wait_complete_ns << ',' << record.sample.yield_count << ','
             << record.sample.wait_progress_submit_count << ',' << record.sample.gpu_idle_overlap_ns << '\n';
    }
#endif
    return path;
}

} // namespace Common::PerformanceTelemetry
