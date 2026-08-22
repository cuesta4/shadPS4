// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <optional>

#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "common/performance_telemetry.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/gpu_authority_tracker.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/aliasing.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

struct PendingImageReadback {
    struct Page {
        VAddr address;
        u64 generation;
    };

    VAddr address{};
    u8* data{};
    Buffer* buffer{};
    u64 buffer_offset{};
    u64 size{};
    boost::container::small_vector<Page, 4> pages;
    bool tracked{};
};

class ReadbackTracker {
public:
    explicit ReadbackTracker(PageManager& page_manager_) : page_manager{&page_manager_} {}

    bool TrackAsync(PendingImageReadback& pending) {
        std::scoped_lock lock{mutex};
        if (canceled) {
            return false;
        }
        active_downloads.fetch_add(1, std::memory_order_release);
        pending.tracked = true;
        ForEachPage(pending.address, pending.size, [&](VAddr page) {
            PageState& state = pages[page];
            if (!state.watched) {
                page_manager->UpdatePageWatchers<true>(page, PageSize(page));
                state.watched = true;
            }
            ++state.readers;
            pending.pages.push_back({page, state.generation});
        });
        return true;
    }

    void Invalidate(VAddr address, u64 size) {
        if (size == 0 || active_downloads.load(std::memory_order_acquire) == 0) {
            return;
        }
        std::scoped_lock lock{mutex};
        if (canceled) {
            return;
        }
        const VAddr begin = PageManager::GetPageAddr(address);
        const VAddr end = PageManager::GetNextPageAddr(address + size - 1);
        for (auto it = pages.lower_bound(begin); it != pages.end() && it->first < end; ++it) {
            const VAddr page = it->first;
            PageState& state = it->second;
            ++state.generation;
            if (state.watched) {
                page_manager->UpdatePageWatchers<false>(page, PageSize(page));
                state.watched = false;
            }
        }
    }

    u64 CompleteAsync(const PendingImageReadback& pending, bool commit) {
        if (!pending.tracked) {
            return 0;
        }
        SCOPE_EXIT {
            active_downloads.fetch_sub(1, std::memory_order_release);
        };
        std::scoped_lock lock{mutex};
        if (canceled) {
            return 0;
        }
        pending.buffer->Invalidate(pending.buffer_offset, pending.size);
        u64 committed_bytes{};
        if (commit) {
            for (const PendingImageReadback::Page& page : pending.pages) {
                const auto it = pages.find(page.address);
                if (it == pages.end() || it->second.generation != page.generation) {
                    continue;
                }
                const VAddr begin = std::max<VAddr>(pending.address, page.address);
                const VAddr end = std::min<VAddr>(pending.address + pending.size,
                                                  page.address + PageSize(page.address));
                const u64 size = end - begin;
                if (Core::Memory::Instance()->TryWriteBacking(
                        std::bit_cast<u8*>(begin), pending.data + (begin - pending.address), size,
                        Core::MemoryWriteOrigin::GpuCompletion)) {
                    committed_bytes += size;
                }
            }
        }
        Release(pending);
        return committed_bytes;
    }

    void Cancel() {
        std::scoped_lock lock{mutex};
        canceled = true;
        for (const auto& [page, state] : pages) {
            if (state.watched) {
                page_manager->UpdatePageWatchers<false>(page, PageSize(page));
            }
        }
        pages.clear();
    }

private:
    struct PageState {
        u64 generation{};
        u32 readers{};
        bool watched{};
    };

    static VAddr PageSize(VAddr page) {
        return PageManager::GetNextPageAddr(page) - page;
    }

    template <typename Func>
    static void ForEachPage(VAddr address, u64 size, Func&& func) {
        const VAddr end = PageManager::GetNextPageAddr(address + size - 1);
        for (VAddr page = PageManager::GetPageAddr(address); page < end; page += PageSize(page)) {
            func(page);
        }
    }

    void Release(const PendingImageReadback& pending) {
        for (const PendingImageReadback::Page& page : pending.pages) {
            const auto it = pages.find(page.address);
            if (it == pages.end()) {
                continue;
            }
            PageState& state = it->second;
            ASSERT(state.readers > 0);
            if (--state.readers != 0) {
                continue;
            }
            if (state.watched) {
                page_manager->UpdatePageWatchers<false>(page.address, PageSize(page.address));
            }
            pages.erase(it);
        }
    }

    std::mutex mutex;
    std::map<VAddr, PageState> pages;
    PageManager* page_manager;
    std::atomic<u32> active_downloads{};
    bool canceled{};
};

[[nodiscard]] static u64 GetDownloadSize(const ImageInfo& info) {
    return static_cast<u64>(info.pitch) * info.size.height * info.size.depth *
           info.resources.layers * (info.num_bits / 8);
}

[[nodiscard]] static bool Covers(const Extent3D& extent, const ImageInfo& info) {
    return extent.width == info.size.width && extent.height == info.size.height &&
           extent.depth == info.size.depth;
}

[[nodiscard]] static bool CanAlias(const Image& lhs, const Image& rhs) {
    return GetAliasCopyExtent(lhs.info, rhs.info) || GetAliasCopyExtent(rhs.info, lhs.info);
}

static void SetAliasIdentity(ImageId& dst_id, u64& dst_uid, ImageId src_id, u64 src_uid) {
    dst_id = src_id;
    dst_uid = src_uid;
}

template <typename Func>
void TextureCache::ForEachAlias(const Image& image, Func&& func) {
    const u64 page_index = image.info.guest_address >> Traits::PageBits;
    const auto page = page_table.find(page_index);
    if (page == nullptr) {
        return;
    }
    for (const ImageId candidate_id : *page) {
        Image& candidate = slot_images[candidate_id];
        if (candidate.info.guest_address == image.info.guest_address &&
            CanAlias(candidate, image)) {
            func(candidate_id, candidate);
        }
    }
}

static void RecordScheduledImageWriteback(
    Image& image, ImageId image_id, Common::PerformanceTelemetry::WritebackTrigger trigger,
    u32 trigger_control, u32 trigger_data_control, u32 queued_images,
    const ImageTelemetryWritebackState& telemetry_state, u64 backing_image) {
    u32 flags{};
    flags |= static_cast<u32>(telemetry_state.previous_epoch != 0 &&
                              telemetry_state.previous_epoch == telemetry_state.content_epoch) *
             Common::PerformanceTelemetry::WritebackFlagSameEpoch;
    flags |= static_cast<u32>(telemetry_state.previous_backing != 0 &&
                              telemetry_state.previous_backing == backing_image) *
             Common::PerformanceTelemetry::WritebackFlagSameBacking;
    flags |= static_cast<u32>(image.info.props.is_tiled) *
             Common::PerformanceTelemetry::WritebackFlagTiled;
    flags |= static_cast<u32>(image.info.size.width <= 8) *
             Common::PerformanceTelemetry::WritebackFlagNarrow;
    flags |= static_cast<u32>(image.usage.storage) *
             Common::PerformanceTelemetry::WritebackFlagStorage;
    flags |= static_cast<u32>(image.usage.render_target) *
             Common::PerformanceTelemetry::WritebackFlagRenderTarget;
    const u64 download_size = GetDownloadSize(image.info);
    Common::PerformanceTelemetry::RecordWritebackImageEnabled({
        .trigger = trigger,
        .writer = telemetry_state.writer,
        .trigger_control = trigger_control,
        .trigger_data_control = trigger_data_control,
        .image_index = image_id.index,
        .queued_images = queued_images,
        .image_uid = image.image_uid,
        .backing_image = backing_image,
        .guest_address = image.info.guest_address,
        .download_bytes = download_size,
        .content_epoch = telemetry_state.content_epoch,
        .previous_epoch = telemetry_state.previous_epoch,
        .previous_backing = telemetry_state.previous_backing,
        .flags = flags,
    });
    image.TelemetryMarkScheduled(backing_image);
}

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           AmdGpu::Liverpool* liverpool_, BufferCache& buffer_cache_,
                           PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_},
      readback_tracker{std::make_shared<ReadbackTracker>(tracker)},
      blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    pressure_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_PRESSURE_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() {
    readback_tracker->Cancel();
}

void TextureCache::UpdateImage(ImageId image_id) {
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::ImageUpdate>
        duration;
    PrepareImageAccess(image_id, AliasAccess::Read);
}

void TextureCache::PrepareImageAccess(ImageId image_id, AliasAccess access) {
    std::scoped_lock lock{mutex};
    UpdateImageImpl(image_id);
    SynchronizeAlias(image_id);
    if (access == AliasAccess::ReadWrite) {
        PublishAliasWrite(image_id);
    }
}

void TextureCache::UpdateImageImpl(ImageId image_id) {
    Image& image = slot_images[image_id];
    TrackImage(image_id);
    TouchImage(image);
    RefreshImage(image);
}

void TextureCache::SynchronizeAlias(ImageId image_id) {
    Image& dst = slot_images[image_id];
    if (False(dst.flags & ImageFlagBits::Aliased)) {
        return;
    }
    auto state_it = alias_states.find(dst.info.guest_address);
    if (state_it == alias_states.end()) {
        return;
    }
    AliasState& state = state_it.value();
    if (!IsLiveImage(state.backing, state.backing_uid)) {
        if (!IsLiveImage(state.writer, state.writer_uid)) {
            state.ResetAuthority();
            return;
        }
        SetAliasIdentity(state.backing, state.backing_uid, state.writer, state.writer_uid);
        SetAliasIdentity(state.writer, state.writer_uid, {}, 0);
    }

    if (state.writer == image_id && state.writer_uid == dst.image_uid) {
        return;
    }
    if (!CommitAliasWriter(state)) {
        state.ResetAuthority();
        return;
    }

    Image& current = slot_images[state.backing];
    const bool same_image = state.backing == image_id;
    const bool up_to_date = current.alias_generation <= dst.alias_generation;
    const std::optional copy_extent = GetAliasCopyExtent(current.info, dst.info);
    if (same_image || up_to_date || !copy_extent) {
        return;
    }

    CopyAlias(state.backing, image_id, *copy_extent);
    dst.alias_generation = current.alias_generation;
    dst.flags |= ImageFlagBits::GpuModified;
    dst.flags &= ~ImageFlagBits::Dirty;
    const std::optional reverse_extent = GetAliasCopyExtent(dst.info, current.info);
    if (reverse_extent && Covers(*reverse_extent, current.info) &&
        !Covers(*copy_extent, dst.info)) {
        SetAliasIdentity(state.backing, state.backing_uid, image_id, dst.image_uid);
    }
}

bool TextureCache::CommitAliasWriter(AliasState& state) {
    if (!state.writer) {
        return true;
    }
    if (!IsLiveImage(state.writer, state.writer_uid) ||
        !IsLiveImage(state.backing, state.backing_uid)) {
        return false;
    }

    Image& writer = slot_images[state.writer];
    Image& backing = slot_images[state.backing];
    if (state.writer != state.backing && writer.alias_generation > backing.alias_generation) {
        const std::optional extent = GetAliasCopyExtent(writer.info, backing.info);
        if (!extent) {
            return false;
        }
        CopyAlias(state.writer, state.backing, *extent);
        backing.alias_generation = writer.alias_generation;
        backing.flags |= ImageFlagBits::GpuModified;
        backing.flags &= ~ImageFlagBits::Dirty;
    }
    SetAliasIdentity(state.writer, state.writer_uid, {}, 0);
    return true;
}

void TextureCache::CopyAlias(ImageId src_id, ImageId dst_id, const Extent3D& extent) {
    Image& src = slot_images[src_id];
    Image& dst = slot_images[dst_id];
    scheduler.EndRendering();
    auto barriers =
        src.GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                        vk::PipelineStageFlagBits2::eCopy, {});
    const auto dst_barriers =
        dst.GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eCopy, {});
    barriers.insert(barriers.end(), dst_barriers.begin(), dst_barriers.end());
    const auto cmdbuf = scheduler.CommandBuffer();
    if (!barriers.empty()) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });
    }

    const vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = std::min(src.info.resources.layers, dst.info.resources.layers),
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = std::min(src.info.resources.layers, dst.info.resources.layers),
            },
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, extent.depth},
    };
    cmdbuf.copyImage(src.GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst.GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, region);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    const u64 dst_previous_version = dst.content_epoch;
#endif
    dst.MarkWrite(Common::PerformanceTelemetry::ImageWriter::Transfer);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    const u64 copy_width =
        src.info.props.is_block ? Common::DivCeil(extent.width, 4u) : extent.width;
    const u64 copy_height =
        src.info.props.is_block ? Common::DivCeil(extent.height, 4u) : extent.height;
    Common::PerformanceTelemetry::RecordGpuAliasMaterialize({
        .source_resource_id = src.image_uid,
        .source_version = src.content_epoch,
        .dest_resource_id = dst.image_uid,
        .dest_previous_version = dst_previous_version,
        .dest_new_version = dst.content_epoch,
        .guest_addr = dst.info.guest_address,
        .size = copy_width * copy_height * extent.depth *
                std::min(src.info.resources.layers, dst.info.resources.layers) *
                (src.info.num_bits / 8),
        .copy_kind = src.info.pixel_format == dst.info.pixel_format
                         ? Common::PerformanceTelemetry::AliasCopyKind::ImageToImage
                         : Common::PerformanceTelemetry::AliasCopyKind::FormatReinterpret,
        .submit_seq = Common::PerformanceTelemetry::LookupSubmitSeq(scheduler.CurrentTick()),
        .packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq(),
    });
#endif
}

void TextureCache::PublishAliasWrite(ImageId image_id) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    readback_tracker->Invalidate(image.info.guest_address, image.info.guest_size);
    image.alias_generation = ++alias_generation;
    if (False(image.flags & ImageFlagBits::Aliased)) {
        return;
    }

    AliasState& state = alias_states[image.info.guest_address];
    if (IsLiveImage(state.backing, state.backing_uid)) {
        const Image& backing = slot_images[state.backing];
        if (!CanAlias(image, backing)) {
            state.ResetAuthority();
        }
    }
    state.members = std::max(state.members, 2u);
    const bool continuing_write = state.writer == image_id && state.writer_uid == image.image_uid;
    if (!state.backing) {
        SetAliasIdentity(state.backing, state.backing_uid, image_id, image.image_uid);
    } else if (!continuing_write && IsLiveImage(state.backing, state.backing_uid)) {
        const Image& backing = slot_images[state.backing];
        const std::optional extent = GetAliasCopyExtent(image.info, backing.info);
        if (extent && Covers(*extent, backing.info)) {
            SetAliasIdentity(state.backing, state.backing_uid, image_id, image.image_uid);
        } else {
            SetAliasIdentity(state.writer, state.writer_uid, image_id, image.image_uid);
        }
    } else if (!continuing_write) {
        SetAliasIdentity(state.backing, state.backing_uid, image_id, image.image_uid);
    }

    const bool needs_download =
        state.members > 1 && GetDownloadSize(image.info) <= TRACKER_BYTES_PER_PAGE;
    if (needs_download && !state.download) {
        pending_alias_downloads.push_back(image.info.guest_address);
    }
    SetAliasIdentity(state.download, state.download_uid, needs_download ? image_id : ImageId{},
                     needs_download ? image.image_uid : 0);
}

bool TextureCache::ProcessDownloadImages(const DownloadContext& context, bool* gpu_resident) {
    std::scoped_lock texture_lock{mutex};
    std::unique_lock downloads_lock{download_images_mutex};
    for (const VAddr address : pending_alias_downloads) {
        const auto state_it = alias_states.find(address);
        if (state_it == alias_states.end()) {
            continue;
        }
        AliasState& state = state_it.value();
        const ImageId download_id = state.download;
        const u64 download_uid = state.download_uid;
        SetAliasIdentity(state.download, state.download_uid, {}, 0);
        if (!IsLiveImage(download_id, download_uid)) {
            continue;
        }
        const Image& image = slot_images[download_id];
        if (image.info.guest_address != address ||
            False(image.flags & ImageFlagBits::GpuModified)) {
            continue;
        }
        const PendingImageDownload download{
            .pending_seq = next_pending_download_seq++,
            .image_id = download_id,
            .image_uid = download_uid,
            .resource_id = download_uid,
            .resource_version = image.content_epoch,
            .guest_begin = address,
            .size = static_cast<u32>(GetDownloadSize(image.info)),
            .policy = DownloadPolicy::LegacyEager,
        };
        std::erase_if(pending_downloads,
                      [address](const auto& entry) { return entry.guest_begin == address; });
        pending_downloads.push_back(download);
    }
    pending_alias_downloads.clear();
    if (pending_downloads.empty()) {
        if (gpu_resident != nullptr) {
            *gpu_resident = false;
        }
        return false;
    }
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    const u32 candidates = static_cast<u32>(pending_downloads.size());
    u32 scheduled_count{};
    bool all_gpu_resident = (context.trigger == DownloadTrigger::EventWriteEos);
    scheduler.EndRendering();

    std::vector<PendingImageDownload> remaining_downloads;
    remaining_downloads.reserve(pending_downloads.size());

    for (const auto& entry : pending_downloads) {
        if (!slot_images.is_allocated(entry.image_id) ||
            slot_images[entry.image_id].image_uid != entry.image_uid) {
            // Stale or freed image
            continue;
        }
        auto& image = slot_images[entry.image_id];
        auto auth = VideoCore::GpuAuthorityTracker::Instance().GetAuthorityForImage(
            entry.image_uid, entry.resource_version);

        if (entry.policy == DownloadPolicy::AuthorityManaged || auth != nullptr) {
            u8 auth_state{};
            if (auth) {
                std::scoped_lock auth_lock{*auth->entry_mutex};
                auth_state = static_cast<u8>(auth->state);
            }
            const u64 auth_seq = auth ? auth->authority_seq : 0;

            if (context.trigger == DownloadTrigger::EventWriteEos ||
                context.trigger == DownloadTrigger::EventWriteEop ||
                context.trigger == DownloadTrigger::ReleaseMem ||
                context.trigger == DownloadTrigger::Other) {
                // Suppress conservative drain!
                Common::PerformanceTelemetry::RecordConservativeDownloadDecision(
                    Common::PerformanceTelemetry::ConservativeDownloadDecisionSample{
                        .timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
                        .trigger = static_cast<u32>(context.trigger),
                        .fence_seq = context.fence_seq,
                        .image_id = entry.image_id.index,
                        .image_uid = entry.image_uid,
                        .resource_id = entry.resource_id,
                        .resource_version = entry.resource_version,
                        .guest_addr = entry.guest_begin,
                        .size = entry.size,
                        .authority_seq = auth_seq,
                        .authority_state = auth_state,
                        .decision = Common::PerformanceTelemetry::ConservativeDownloadDecision::SuppressAuthority,
                        .reason = 0,
                        .readback_schedule_seen = 0,
                        .readback_seq = 0,
                    });
                Common::PerformanceTelemetry::RecordAuthorityConservativeReadbackSuppressed(
                    Common::PerformanceTelemetry::AuthorityConservativeReadbackSuppressedSample{
                        .authority_seq = auth_seq,
                        .resource_id = entry.resource_id,
                        .resource_version = entry.resource_version,
                        .trigger = static_cast<u32>(context.trigger),
                        .fence_seq = context.fence_seq,
                        .guest_addr = entry.guest_begin,
                        .size = entry.size,
                        .image_id = entry.image_id.index,
                        .image_uid = entry.image_uid,
                    });
                Common::PerformanceTelemetry::Add(
                    Common::PerformanceTelemetry::Counter::ConservativeDownloadSuppressed);
                Common::PerformanceTelemetry::Add(
                    Common::PerformanceTelemetry::Counter::Eager3kDownloads);
                remaining_downloads.push_back(entry);
                continue;
            }
        }

        // Legacy eager download
        ImageTelemetryWritebackState telemetry_state{};
        u64 backing_image{};
        if (telemetry_enabled) {
            telemetry_state = image.TelemetryWritebackState();
            backing_image = std::bit_cast<u64>(static_cast<VkImage>(image.GetImage()));
        }
        bool image_gpu_resident{};
        if (!DownloadImageMemory(entry.image_id, true, all_gpu_resident, &image_gpu_resident)) {
            continue;
        }
        ++scheduled_count;
        all_gpu_resident &= image_gpu_resident;
        if (telemetry_enabled) {
            const auto wb_trigger = static_cast<Common::PerformanceTelemetry::WritebackTrigger>(context.trigger);
            RecordScheduledImageWriteback(image, entry.image_id, wb_trigger, context.trigger_control,
                                          context.trigger_data_control, candidates, telemetry_state,
                                          backing_image);
            Common::PerformanceTelemetry::RecordConservativeDownloadDecision(
                Common::PerformanceTelemetry::ConservativeDownloadDecisionSample{
                    .timestamp_ns = Common::PerformanceTelemetry::Timestamp(),
                    .trigger = static_cast<u32>(context.trigger),
                    .fence_seq = context.fence_seq,
                    .image_id = entry.image_id.index,
                    .image_uid = entry.image_uid,
                    .resource_id = entry.resource_id,
                    .resource_version = entry.resource_version,
                    .guest_addr = entry.guest_begin,
                    .size = entry.size,
                    .authority_seq = 0,
                    .authority_state = 0,
                    .decision = Common::PerformanceTelemetry::ConservativeDownloadDecision::LegacyDownload,
                    .reason = 0,
                    .readback_schedule_seen = 1,
                    .readback_seq = 0,
                });
        }
    }
    pending_downloads = std::move(remaining_downloads);

    if (telemetry_enabled) {
        const auto wb_trigger = static_cast<Common::PerformanceTelemetry::WritebackTrigger>(context.trigger);
        Common::PerformanceTelemetry::RecordWritebackDrainEnabled(wb_trigger, candidates,
                                                                  scheduled_count);
    }
    if (scheduled_count != 0) {
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::WritebackBatches);
    }
    if (gpu_resident != nullptr) {
        *gpu_resident = scheduled_count != 0 && all_gpu_resident;
    }
    return scheduled_count != 0;
}

bool TextureCache::ProcessDownloadImages(Common::PerformanceTelemetry::WritebackTrigger trigger,
                                         u32 trigger_control, u32 trigger_data_control,
                                         bool* gpu_resident) {
    DownloadTrigger dl_trigger{DownloadTrigger::Other};
    switch (trigger) {
    case Common::PerformanceTelemetry::WritebackTrigger::EventWriteEos:
        dl_trigger = DownloadTrigger::EventWriteEos;
        break;
    case Common::PerformanceTelemetry::WritebackTrigger::EventWriteEop:
        dl_trigger = DownloadTrigger::EventWriteEop;
        break;
    case Common::PerformanceTelemetry::WritebackTrigger::ReleaseMem:
        dl_trigger = DownloadTrigger::ReleaseMem;
        break;
    default:
        dl_trigger = DownloadTrigger::Other;
        break;
    }
    return ProcessDownloadImages(
        DownloadContext{
            .trigger = dl_trigger,
            .fence_seq = 0,
            .trigger_control = trigger_control,
            .trigger_data_control = trigger_data_control,
        },
        gpu_resident);
}

bool TextureCache::PromotePendingDownloadAuthority(ImageId image_id, u64 image_uid,
                                                   u64 resource_version,
                                                   std::shared_ptr<GpuAuthorityShadow>* shadow) {
    if (shadow == nullptr) {
        return false;
    }
    shadow->reset();
    std::scoped_lock texture_lock{mutex};
    std::unique_lock downloads_lock{download_images_mutex};
    if (pending_downloads.size() != 1) {
        return false;
    }
    for (auto it = pending_downloads.begin(); it != pending_downloads.end(); ++it) {
        auto& entry = *it;
        if (entry.image_id == image_id && entry.image_uid == image_uid &&
            entry.resource_version == resource_version) {
            if (!slot_images.is_allocated(image_id)) {
                return false;
            }
            auto& image = slot_images[image_id];
            if (True(image.flags & ImageFlagBits::Aliased)) {
                const auto state_it = alias_states.find(image.info.guest_address);
                if (state_it != alias_states.end()) {
                    const AliasState& state = state_it.value();
                    if (state.download &&
                        (state.download != image_id || state.download_uid != image_uid)) {
                        return false;
                    }
                }
            }
            const u32 download_size = static_cast<u32>(GetDownloadSize(image.info));
            auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
            if (image.image_uid != image_uid || image.content_epoch != resource_version ||
                !image.SafeToDownload() || download_size != entry.size ||
                download_size > download_buffer.SizeBytes() ||
                !buffer_cache.TrackImageReadback(image, entry.size)) {
                return false;
            }

            const auto [download, offset] = download_buffer.Map(download_size);
            ASSERT(download != nullptr);
            auto authority_shadow = std::make_shared<GpuAuthorityShadow>(
                download, image.info.guest_address, offset, download_size,
                scheduler.CurrentTick());
            download_buffer.Commit(authority_shadow);
            const vk::BufferImageCopy image_download = {
                .bufferOffset = offset,
                .bufferRowLength = image.info.pitch,
                .bufferImageHeight = image.info.size.height,
                .imageSubresource =
                    {
                        .aspectMask = image.info.props.is_depth
                                          ? vk::ImageAspectFlagBits::eDepth
                                          : vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = image.info.resources.layers,
                    },
                .imageOffset = {0, 0, 0},
                .imageExtent = {image.info.size.width, image.info.size.height,
                                image.info.size.depth},
            };
            image.Transit(vk::ImageLayout::eTransferSrcOptimal,
                          vk::AccessFlagBits2::eTransferRead, {});
            scheduler.CommandBuffer().copyImageToBuffer(
                image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                download_buffer.Handle(), image_download);
            image.flags &= ~ImageFlagBits::GpuModified;
            Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyCalls);
            Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyBytes,
                                              download_size);
            *shadow = std::move(authority_shadow);
            pending_downloads.erase(it);
            if (True(image.flags & ImageFlagBits::Aliased)) {
                const auto state_it = alias_states.find(image.info.guest_address);
                if (state_it != alias_states.end()) {
                    AliasState& state = state_it.value();
                    if (state.download == image_id && state.download_uid == image_uid) {
                        SetAliasIdentity(state.download, state.download_uid, {}, 0);
                        const auto address = image.info.guest_address;
                        pending_alias_downloads.erase(
                            std::remove(pending_alias_downloads.begin(),
                                        pending_alias_downloads.end(), address),
                            pending_alias_downloads.end());
                    }
                }
            }
            return true;
        }
    }
    return false;
}

void TextureCache::PruneSupersededPendingDownloads(u64 image_uid, u64 superseded_version) {
    std::unique_lock lk{download_images_mutex};
    std::erase_if(pending_downloads, [image_uid, superseded_version](const PendingImageDownload& d) {
        return d.image_uid == image_uid && d.resource_version <= superseded_version;
    });
}

bool TextureCache::DownloadImageMemory(ImageId image_id, bool validate_identity,
                                       bool track_gpu_source, bool* gpu_resident) {
    if (gpu_resident != nullptr) {
        *gpu_resident = false;
    }
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return false;
    }
    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    const u32 download_size = static_cast<u32>(GetDownloadSize(image.info));
    ASSERT(download_size <= image.info.guest_size);
    const u64 writeback_start = Common::PerformanceTelemetry::Enabled()
                                    ? Common::PerformanceTelemetry::Timestamp()
                                    : 0;
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::WritebackCalls);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::WritebackBytes,
                                      download_size);
    const auto [download, offset] = download_buffer.Map(download_size);
    download_buffer.Commit();
    const vk::BufferImageCopy image_download = {
        .bufferOffset = offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    const auto cmdbuf = scheduler.CommandBuffer();
    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyCalls);
    Common::PerformanceTelemetry::Add(Common::PerformanceTelemetry::Counter::CopyBytes,
                                      download_size);
    cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             download_buffer.Handle(), image_download);
    image.flags &= ~ImageFlagBits::GpuModified;

    if (track_gpu_source && gpu_resident != nullptr) {
        *gpu_resident = buffer_cache.TrackImageReadback(image, download_size);
    }

    auto readback_token = validate_identity ? image.readback_token : nullptr;
    const u64 image_uid = image.image_uid;
    PendingImageReadback pending{
        .address = image.info.guest_address,
        .data = download,
        .buffer = &download_buffer,
        .buffer_offset = offset,
        .size = download_size,
    };
    readback_tracker->TrackAsync(pending);
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
    const u64 image_content_epoch = image.content_epoch;
    const auto readback_seq = writeback_start != 0 ? Common::PerformanceTelemetry::NextReadbackSeq() : 0;
    const u64 sched_tick = scheduler.CurrentTick();
    const auto active_fence_cause = Common::PerformanceTelemetry::CurrentFenceCause();
    const auto submit_seq = Common::PerformanceTelemetry::LookupSubmitSeq(sched_tick);
    const auto cur_cmdbuf = Common::PerformanceTelemetry::CurrentCmdBufferSeq();
    if (writeback_start != 0) {
        Common::PerformanceTelemetry::RecordReadbackSchedule(Common::PerformanceTelemetry::ReadbackScheduleSample{
            .readback_seq = readback_seq,
            .fence_seq = active_fence_cause.fence_seq,
            .resource_id = image.image_uid,
            .version = image_content_epoch,
            .guest_addr = image.info.guest_address,
            .size = download_size,
            .download_offset = offset,
            .producer_tick = sched_tick,
            .schedule_tick = sched_tick,
            .reason = Common::PerformanceTelemetry::ReadbackReason::FenceConservative,
            .correlation_status = active_fence_cause.fence_seq != 0 ? Common::PerformanceTelemetry::CorrelationStatus::Complete : Common::PerformanceTelemetry::CorrelationStatus::MissingContext,
        });
        Common::PerformanceTelemetry::RecordReadbackSubmit(Common::PerformanceTelemetry::ReadbackSubmitSample{
            .readback_seq = readback_seq,
            .submit_seq = submit_seq,
            .ready_tick = sched_tick,
            .enqueue_ns = writeback_start,
            .copy_bytes = download_size,
            .cmd_buffer_seq = cur_cmdbuf,
            .signal_tick = 0,
        });
        Common::PerformanceTelemetry::ArmReadbackSourceWatch(Common::PerformanceTelemetry::ReadbackSourceWatch{
            .watch_seq = Common::PerformanceTelemetry::NextWatchSeq(),
            .fence_seq = active_fence_cause.fence_seq,
            .wait_seq = 0,
            .readback_seq = readback_seq,
            .producer_seq = 0,
            .resource_id = image.image_uid,
            .resource_version = image_content_epoch,
            .guest_addr = image.info.guest_address,
            .size = download_size,
            .producer_packet = 0,
            .fence_packet = active_fence_cause.packet_seq,
            .state = Common::PerformanceTelemetry::SourceWatchState::Active,
            .create_timestamp_ns = writeback_start,
        });
        Common::PerformanceTelemetry::RegisterPendingReadbackForSubmit(readback_seq, cur_cmdbuf);
    }
    const Common::PerformanceTelemetry::PendingOpTraceToken pending_trace{
        .fence_seq = active_fence_cause.fence_seq,
        .readback_seq = readback_seq,
        .submit_seq = submit_seq,
        .packet_seq = active_fence_cause.packet_seq != 0 ? active_fence_cause.packet_seq : Common::PerformanceTelemetry::CurrentPacketSeq(),
    };
#else
    const u64 image_content_epoch = 0;
    const auto readback_seq = 0;
    const Common::PerformanceTelemetry::PendingOpTraceToken pending_trace{};
#endif

    scheduler.DeferPriorityOperation(
        [device_addr = image.info.guest_address, download_size, writeback_start,
         image_uid, image_content_epoch, readback_seq, pending_trace,
         readback_token = std::move(readback_token), pending = std::move(pending),
         readback_tracker = readback_tracker, &page_manager = tracker] {
            const auto record_completion = [=] {
                if (writeback_start != 0) {
                    Common::PerformanceTelemetry::RecordDurationEnabled(
                        Common::PerformanceTelemetry::Counter::WritebackNs,
                        Common::PerformanceTelemetry::EventType::Writeback, writeback_start,
                        download_size);
                }
            };
            const auto write_back = [&] {
                Common::PerformanceTelemetry::ScopedMemoryWriteOrigin origin{
                    Common::PerformanceTelemetry::MemoryWriteOrigin::ReadbackCommit,
                    pending_trace.fence_seq};
                return readback_tracker->CompleteAsync(pending, true);
            };
            if (writeback_start != 0) {
                const u64 ready_time = Common::PerformanceTelemetry::Timestamp();
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                Common::PerformanceTelemetry::RecordReadbackReady(Common::PerformanceTelemetry::ReadbackReadySample{
                    .readback_seq = readback_seq,
                    .submit_seq = pending_trace.submit_seq,
                    .ready_tick = 0,
                    .schedule_to_ready_ns = ready_time > writeback_start ? ready_time - writeback_start : 0,
                    .submit_to_ready_ns = ready_time > writeback_start ? ready_time - writeback_start : 0,
                });
#endif
            }
            if (readback_token) {
                std::scoped_lock identity_lock{readback_token->mutex};
                if (readback_token->image_uid != image_uid) {
                    readback_tracker->CompleteAsync(pending, false);
                    Common::PerformanceTelemetry::Add(
                        Common::PerformanceTelemetry::Counter::WritebackStaleSkips);
                    record_completion();
                    return;
                }
                const u64 commit_start = Common::PerformanceTelemetry::Timestamp();
                const u64 committed_bytes = write_back();
                const u64 commit_end = Common::PerformanceTelemetry::Timestamp();
                if (committed_bytes != download_size) {
                    Common::PerformanceTelemetry::Add(
                        Common::PerformanceTelemetry::Counter::WritebackStaleSkips);
                }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (writeback_start != 0) {
                    Common::PerformanceTelemetry::RecordReadbackCommit(Common::PerformanceTelemetry::ReadbackCommitSample{
                        .readback_seq = readback_seq,
                        .fence_seq = pending_trace.fence_seq,
                        .resource_id = image_uid,
                        .version = image_content_epoch,
                        .guest_addr = device_addr,
                        .size = download_size,
                        .memcpy_ns = commit_end - commit_start,
                        .invalidate_ns = 0,
                        .host_version_before = image_content_epoch > 0 ? image_content_epoch - 1 : 0,
                        .host_version_after = image_content_epoch,
                    });
                    Common::PerformanceTelemetry::UpdateHostVersion(
                        device_addr, download_size, image_content_epoch,
                        Common::PerformanceTelemetry::HostVersionOrigin::ReadbackCommit,
                        readback_seq, image_uid);
                    if (!Common::PerformanceTelemetry::IsSyncSemanticProfile() || download_size == 3072) {
                        Common::PerformanceTelemetry::ArmReadWatchInterest(Common::PerformanceTelemetry::ReadWatchInterest{
                            .fence_seq = pending_trace.fence_seq,
                            .generation = 0,
                            .readback_seq = readback_seq,
                            .resource_id = image_uid,
                            .resource_version = image_content_epoch,
                            .guest_addr = device_addr,
                            .size = download_size,
                            .kind = Common::PerformanceTelemetry::ReadWatchKind::Data,
                            .fence_packet = pending_trace.packet_seq,
                            .arm_timestamp_ns = commit_start,
                        });
                        if (Common::PerformanceTelemetry::IsSyncSemanticProfile() && download_size == 3072) {
                            page_manager.UpdatePageWatchers<true, true>(device_addr, download_size);
                        }
                    }
                }
#endif
            } else {
                const u64 commit_start = Common::PerformanceTelemetry::Timestamp();
                const u64 committed_bytes = write_back();
                const u64 commit_end = Common::PerformanceTelemetry::Timestamp();
                if (committed_bytes != download_size) {
                    Common::PerformanceTelemetry::Add(
                        Common::PerformanceTelemetry::Counter::WritebackStaleSkips);
                }
#ifdef SHADPS4_ENABLE_DETAILED_TELEMETRY
                if (writeback_start != 0) {
                    Common::PerformanceTelemetry::RecordReadbackCommit(Common::PerformanceTelemetry::ReadbackCommitSample{
                        .readback_seq = readback_seq,
                        .fence_seq = pending_trace.fence_seq,
                        .resource_id = image_uid,
                        .version = image_content_epoch,
                        .guest_addr = device_addr,
                        .size = download_size,
                        .memcpy_ns = commit_end - commit_start,
                        .invalidate_ns = 0,
                        .host_version_before = image_content_epoch > 0 ? image_content_epoch - 1 : 0,
                        .host_version_after = image_content_epoch,
                    });
                    Common::PerformanceTelemetry::UpdateHostVersion(
                        device_addr, download_size, image_content_epoch,
                        Common::PerformanceTelemetry::HostVersionOrigin::ReadbackCommit,
                        readback_seq, image_uid);
                    if (!Common::PerformanceTelemetry::IsSyncSemanticProfile() || download_size == 3072) {
                        Common::PerformanceTelemetry::ArmReadWatchInterest(Common::PerformanceTelemetry::ReadWatchInterest{
                            .fence_seq = pending_trace.fence_seq,
                            .generation = 0,
                            .readback_seq = readback_seq,
                            .resource_id = image_uid,
                            .resource_version = image_content_epoch,
                            .guest_addr = device_addr,
                            .size = download_size,
                            .kind = Common::PerformanceTelemetry::ReadWatchKind::Data,
                            .fence_packet = pending_trace.packet_seq,
                            .arm_timestamp_ns = commit_start,
                        });
                        if (Common::PerformanceTelemetry::IsSyncSemanticProfile() && download_size == 3072) {
                            page_manager.UpdatePageWatchers<true, true>(device_addr, download_size);
                        }
                    }
                }
#endif
            }
            record_completion();
        },
        pending_trace);
    if (writeback_start != 0) {
        Common::PerformanceTelemetry::RecordDurationEnabled(
            Common::PerformanceTelemetry::Counter::WritebackEnqueueNs,
            Common::PerformanceTelemetry::EventType::None, writeback_start, download_size);
    }
    return true;
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateAlias(Image& image) {
    if (False(image.flags & ImageFlagBits::Aliased)) {
        image.alias_generation = 0;
        return;
    }
    if (auto state_it = alias_states.find(image.info.guest_address);
        state_it != alias_states.end()) {
        state_it.value().ResetAuthority();
    }
    image.alias_generation = 0;
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    readback_tracker->Invalidate(addr, size);
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            InvalidateAlias(image);
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    std::scoped_lock lock{mutex};
    readback_tracker->Invalidate(address, max_size);
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        InvalidateAlias(image);
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};
    readback_tracker->Invalidate(cpu_addr, size);

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        const auto new_image_id =
            slot_images.insert(instance, scheduler, blit_helper, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;

        if (cache_image.info.num_samples == 1 && new_info.num_samples == 1) {
            // Perform depth<->color copy using the intermediate copy buffer.
            if (instance.IsMaintenance8Supported()) {
                new_image.CopyImage(cache_image);
            } else {
                const auto& copy_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
                new_image.CopyImageWithBuffer(cache_image, copy_buffer.Handle(), 0);
            }
        } else if (cache_image.info.num_samples == 1 && new_info.props.is_depth &&
                   new_info.num_samples > 1) {
            // Perform a rendering pass to transfer the channels of source as samples in dest.
            cache_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::AccessFlagBits2::eShaderRead, {});
            new_image.Transit(vk::ImageLayout::eDepthAttachmentOptimal,
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {});
            blit_helper.ReinterpretColorAsMsDepth(
                new_info.size.width, new_info.size.height, new_info.num_samples,
                cache_image.info.pixel_format, new_info.pixel_format, cache_image.GetImage(),
                new_image.GetImage());
        } else {
            LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
        }

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    merged_image.CopyMip(cache_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    new_image.CopyImage(src_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);

    TrackImage(new_image_id);
    new_image.flags &= ~ImageFlagBits::Dirty;
    return new_image_id;
}

bool TextureCache::TryReuseImage(ImageId image_id, u64 image_uid, u64 expected_topology_epoch) {
    std::scoped_lock lock{mutex};
    if (expected_topology_epoch != topology_epoch.load(std::memory_order_relaxed)) {
        return false;
    }
    if (!image_id || !slot_images.is_allocated(image_id)) {
        return false;
    }
    auto& image = slot_images[image_id];
    if (image.image_uid != image_uid || False(image.flags & ImageFlagBits::Registered)) {
        return false;
    }
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);
    return true;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::ImageFind>
        duration{telemetry_enabled};
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    std::scoped_lock lock{mutex};

    const ExactImageCacheKey exact_key{{
        info.guest_address,
        static_cast<u64>(info.guest_size) | (static_cast<u64>(info.size.width) << 32),
        static_cast<u64>(info.size.height) | (static_cast<u64>(info.size.depth) << 32),
        static_cast<u64>(info.resources.levels) |
            (static_cast<u64>(info.resources.layers) << 32),
        static_cast<u32>(info.pixel_format) | (static_cast<u64>(exact_fmt) << 32),
        static_cast<u64>(info.type),
    }};
    const auto hash_key = [&] {
        u64 value = info.guest_address ^ (static_cast<u64>(info.guest_size) << 17);
        value ^= static_cast<u64>(info.size.width) << 1;
        value ^= static_cast<u64>(info.size.height) << 21;
        value ^= static_cast<u64>(info.size.depth) << 41;
        value ^= static_cast<u64>(info.resources.levels) << 9;
        value ^= static_cast<u64>(info.resources.layers) << 29;
        value ^= static_cast<u64>(info.pixel_format) << 45;
        value ^= static_cast<u64>(info.type) * 0x9E3779B185EBCA87ULL;
        value ^= static_cast<u64>(exact_fmt) << 63;
        value ^= value >> 29;
        value *= 0x9E3779B185EBCA87ULL;
        value ^= value >> 32;
        return static_cast<size_t>(value) & (ExactImageCacheSize - 1);
    };
    const auto is_perfect_match = [&](const ImageInfo& cached_info) {
        return cached_info.guest_address == info.guest_address &&
               cached_info.guest_size == info.guest_size && cached_info.size == info.size &&
               IsVulkanFormatCompatible(cached_info.pixel_format, info.pixel_format) &&
               (cached_info.type == info.type || info.size == Extent3D{1, 1, 1}) &&
               (!exact_fmt || cached_info.pixel_format == info.pixel_format) &&
               !(cached_info.resources < info.resources);
    };

    auto& exact_entry = exact_image_cache[hash_key()];
    const u64 current_topology_epoch = topology_epoch.load(std::memory_order_relaxed);
    if (exact_entry.valid &&
        std::memcmp(exact_entry.key.words.data(), exact_key.words.data(),
                    sizeof(ExactImageCacheKey)) == 0 &&
        exact_entry.topology_epoch == current_topology_epoch) {
        if (exact_entry.image_id && slot_images.is_allocated(exact_entry.image_id)) {
            auto& cached_image = slot_images[exact_entry.image_id];
            if (cached_image.image_uid == exact_entry.image_uid &&
                True(cached_image.flags & ImageFlagBits::Registered) &&
                is_perfect_match(cached_image.info)) {
                cached_image.tick_accessed_last = scheduler.CurrentTick();
                TouchImage(cached_image);
                if (telemetry_enabled) {
                    Common::PerformanceTelemetry::RecordImageFindPathEnabled(
                        Common::PerformanceTelemetry::ImageFindPath::ExactCache);
                }
                return exact_entry.image_id;
            }
        }
        exact_entry.valid = false;
    }

    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};
    auto find_path = Common::PerformanceTelemetry::ImageFindPath::PerfectScan;

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        find_path = Common::PerformanceTelemetry::ImageFindPath::Overlap;
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        find_path = Common::PerformanceTelemetry::ImageFindPath::Created;
        image_id = slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    if (view_mip < 0 && view_slice < 0 && is_perfect_match(image.info)) {
        exact_entry = {
            .key = exact_key,
            .image_id = image_id,
            .image_uid = image.image_uid,
            .topology_epoch = topology_epoch.load(std::memory_order_relaxed),
            .valid = true,
        };
    }

    if (telemetry_enabled) {
        Common::PerformanceTelemetry::RecordImageFindPathEnabled(find_path);
    }
    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageId TextureCache::FindImageContainingRange(VAddr address, size_t size) {
    if (size == 0) {
        return {};
    }
    ImageId result{};
    u32 result_size = std::numeric_limits<u32>::max();
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        const VAddr image_addr = image.info.guest_address;
        if (!image.SafeToDownload() || address < image_addr || size > image.info.guest_size ||
            address - image_addr > image.info.guest_size - size) {
            return;
        }
        if (image.info.guest_size < result_size) {
            result = image_id;
            result_size = image.info.guest_size;
        }
    });
    return result;
}

void TextureCache::PrepareTexture(ImageId image_id, const ImageDesc& desc, bool is_compute) {
    static_cast<void>(is_compute);
    const bool is_storage = desc.type == BindingType::Storage;
    PrepareImageAccess(image_id, is_storage ? AliasAccess::ReadWrite : AliasAccess::Read);
}

void TextureCache::ScheduleComputeDownload(ImageId image_id) {
    ScheduleImageDownload(image_id, true);
}

void TextureCache::ScheduleRenderTargetDownload(ImageId image_id) {
    ScheduleImageDownload(image_id, false);
}

void TextureCache::ScheduleImageDownload(ImageId image_id, bool fastpath_candidate) {
    Image& image = slot_images[image_id];
    if (!readback_linear_images || (image.info.props.is_tiled && image.info.size.width > 8) ||
        image.info.guest_address == 0) {
        return;
    }
    const u32 download_size = static_cast<u32>(GetDownloadSize(image.info));
    std::unique_lock candidate_lock{fastpath_candidate_mutex, std::defer_lock};
    if (fastpath_candidate) {
        candidate_lock.lock();
        pending_fastpath_candidate = PendingFastpathCandidate{
            .image_id = image_id,
            .image_uid = image.image_uid,
            .resource_version = image.content_epoch,
            .guest_addr = image.info.guest_address,
            .download_size = download_size,
            .producer_seq = Common::PerformanceTelemetry::CurrentProducerSeq(),
            .producer_packet_seq = Common::PerformanceTelemetry::CurrentPacketSeq(),
        };
    }
    std::unique_lock downloads_lock{download_images_mutex};
    if (!fastpath_candidate) {
        std::erase_if(pending_downloads, [image_id, image_uid = image.image_uid](const auto& entry) {
            return entry.image_id == image_id && entry.image_uid == image_uid;
        });
    }
    pending_downloads.push_back(PendingImageDownload{
        .pending_seq = next_pending_download_seq++,
        .image_id = image_id,
        .image_uid = image.image_uid,
        .resource_id = image.image_uid,
        .resource_version = image.content_epoch,
        .guest_begin = image.info.guest_address,
        .size = download_size,
        .policy = DownloadPolicy::LegacyEager,
    });
    Common::PerformanceTelemetry::Add(
        Common::PerformanceTelemetry::Counter::ConservativeDownloadConsidered);
}

std::optional<TextureCache::PendingFastpathCandidate> TextureCache::TakePendingFastpathCandidate() {
    std::scoped_lock lk{fastpath_candidate_mutex};
    auto candidate = pending_fastpath_candidate;
    pending_fastpath_candidate.reset();
    return candidate;
}

bool TextureCache::IsGpuAuthorityImageCurrent(ImageId image_id, u64 image_uid,
                                              u64 resource_version, VAddr address, size_t size) {
    std::scoped_lock lock{mutex};
    if (!image_id || !slot_images.is_allocated(image_id)) {
        return false;
    }
    const auto& image = slot_images[image_id];
    const VAddr image_addr = image.info.guest_address;
    return image.image_uid == image_uid && image.content_epoch == resource_version &&
           image.SafeToDownload() && address >= image_addr && size <= image.info.guest_size &&
           address - image_addr <= image.info.guest_size - size;
}

void TextureCache::WaitGpuAuthorityShadow(
    const std::shared_ptr<GpuAuthorityShadow>& shadow) {
    if (!shadow) {
        return;
    }
    auto* master_semaphore = scheduler.GetMasterSemaphore();
    if (!master_semaphore->IsFree(shadow->tick) && liverpool->IsGpuThread() &&
        shadow->tick >= scheduler.CurrentTick()) {
        scheduler.Flush(Common::PerformanceTelemetry::SubmitReason::WaitProgress);
    }
    master_semaphore->Wait(shadow->tick,
                           Common::PerformanceTelemetry::HostWaitReason::FenceCpuVisibility);
}

bool TextureCache::MaterializeGpuAuthority(
    const std::shared_ptr<GpuAuthorityShadow>& shadow, VAddr required_addr, size_t required_size,
    s8* out_validation_bytes_equal) {
    if (out_validation_bytes_equal != nullptr) {
        *out_validation_bytes_equal = -1;
    }
    if (!shadow || shadow->guest_addr != required_addr || shadow->size != required_size ||
        !scheduler.GetMasterSemaphore()->IsFree(shadow->tick)) {
        return false;
    }
    std::scoped_lock shadow_lock{shadow->data_mutex};
    if (shadow->data == nullptr) {
        return false;
    }

    const bool result = Core::Memory::Instance()->TryWriteBacking(
        std::bit_cast<u8*>(required_addr), shadow->data, required_size,
        Core::MemoryWriteOrigin::GpuCompletion);
    if (result) {
        const auto complete_readback = [this, required_addr, required_size] {
            buffer_cache.CompleteImageReadback(required_addr,
                                               static_cast<u32>(required_size));
        };
        if (liverpool->IsGpuThread()) {
            complete_readback();
        } else {
            liverpool->SendCommand([complete_readback, required_addr, required_size] {
                if (GpuAuthorityTracker::Instance()
                        .FindOverlaps(required_addr, required_size)
                        .empty()) {
                    complete_readback();
                }
            });
        }
    }
    return result;
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    PrepareTexture(image_id, desc);
    Image& image = slot_images[image_id];
    return image.FindView(desc.view_info);
}

void TextureCache::PrepareRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::RenderTargetPrepare>
        duration;
    Image& image = slot_images[image_id];
    image.usage.render_target = 1u;
    PrepareImageAccess(image_id, AliasAccess::ReadWrite);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaType::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaType::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    PrepareRenderTarget(image_id, desc);
    Image& image = slot_images[image_id];
    return image.FindView(desc.view_info, false);
}

void TextureCache::PrepareDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Common::PerformanceTelemetry::SampledDuration<
        Common::PerformanceTelemetry::TimerSite::RenderTargetPrepare>
        duration;
    Image& image = slot_images[image_id];
    image.usage.depth_target = 1u;
    UpdateImage(image_id);
    readback_tracker->Invalidate(image.info.guest_address, image.info.guest_size);
    image.flags |= ImageFlagBits::GpuModified;

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaType::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        ImageId stencil_id{};
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId image_id, Image& image) {
                                 if (image.info.guest_address == desc.info.stencil_addr) {
                                     stencil_id = image_id;
                                 }
                             });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id =
                slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    PrepareDepthTarget(image_id, desc);
    Image& image = slot_images[image_id];
    return image.FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    if (auto barrier = in_buffer->GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                             vk::PipelineStageFlagBits2::eTransfer)) {
        scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier.value(),
        });
    }

    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer->Handle(), in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    image.Upload(image_copies, buffer, offset);
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base) {
    const u64 hash =
        HashCombine(XXH3_64bits(&sampler, sizeof(sampler)), border_color_base.Address());

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] = samplers.try_emplace(hash, instance, sampler, border_color_base);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    ++topology_epoch;
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    image.lru_tick = gc_tick;

    AliasState* state{};
    bool inserted{};
    u32 aliases{};
    ForEachAlias(image, [&](ImageId candidate_id, Image& candidate) {
        if (!state) {
            auto result = alias_states.try_emplace(image.info.guest_address);
            state = std::addressof(result.first.value());
            inserted = result.second;
        }
        ++aliases;
        candidate.flags |= ImageFlagBits::Aliased;
        if (!IsLiveImage(state->backing, state->backing_uid) ||
            slot_images[state->backing].alias_generation < candidate.alias_generation) {
            SetAliasIdentity(state->backing, state->backing_uid, candidate_id,
                             candidate.image_uid);
        }
    });
    if (state) {
        image.flags |= ImageFlagBits::Aliased;
        state->members = inserted ? aliases + 1 : state->members + 1;
    }

    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    ++topology_epoch;
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(image_begin, size);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(addr, size);
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        auto& image = slot_images[image_id];
        if (auto authority = VideoCore::GpuAuthorityTracker::Instance().GetAuthorityForImage(
                image.image_uid, image.content_epoch)) {
            std::scoped_lock authority_lock{*authority->entry_mutex};
            if (authority->state == GpuAuthorityState::GpuAuthoritative ||
                authority->state == GpuAuthorityState::Materializing) {
                return false;
            }
        }
        --num_deletions;
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        if (download) {
            const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
            ImageTelemetryWritebackState telemetry_state{};
            u64 backing_image{};
            if (telemetry_enabled) {
                telemetry_state = image.TelemetryWritebackState();
                backing_image = std::bit_cast<u64>(static_cast<VkImage>(image.GetImage()));
            }
            const bool scheduled = DownloadImageMemory(image_id);
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::RecordWritebackDrainEnabled(
                    Common::PerformanceTelemetry::WritebackTrigger::GarbageCollection, 1,
                    static_cast<u32>(scheduled));
                if (scheduled) {
                    RecordScheduledImageWriteback(
                        image, image_id,
                        Common::PerformanceTelemetry::WritebackTrigger::GarbageCollection, 0, 0,
                        1, telemetry_state, backing_image);
                }
            }
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const size_t lru_id = samplers.at(hash).lru_id;
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };

    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::TouchImageSlow(Image& image) {
    image.lru_tick = gc_tick;
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    if (image.readback_token) {
        std::scoped_lock lk{image.readback_token->mutex};
        image.readback_token->image_uid = 0;
    }

    const auto alias_it = True(image.flags & ImageFlagBits::Aliased)
                              ? alias_states.find(image.info.guest_address)
                              : alias_states.end();
    if (alias_it != alias_states.end()) {
        AliasState& state = alias_it.value();
        bool state_valid = true;
        if (state.download == image_id && state.download_uid == image.image_uid) {
            SetAliasIdentity(state.download, state.download_uid, {}, 0);
        }
        const bool owns_authority =
            (state.writer == image_id && state.writer_uid == image.image_uid) ||
            (state.backing == image_id && state.backing_uid == image.image_uid);
        if (owns_authority) {
            state_valid = CommitAliasWriter(state);
        }
        if (!state_valid) {
            state.ResetAuthority();
        } else if (state.backing == image_id && state.backing_uid == image.image_uid) {
            ImageId replacement{};
            ForEachAlias(image, [&](ImageId candidate_id, Image& candidate) {
                if (replacement) {
                    return;
                }
                const std::optional to_backing = GetAliasCopyExtent(candidate.info, image.info);
                const std::optional to_candidate = GetAliasCopyExtent(image.info, candidate.info);
                if (to_backing && to_candidate && Covers(*to_backing, image.info) &&
                    Covers(*to_candidate, candidate.info)) {
                    replacement = candidate_id;
                }
            });
            if (replacement) {
                Image& candidate = slot_images[replacement];
                if (candidate.alias_generation < image.alias_generation) {
                    CopyAlias(image_id, replacement, candidate.info.size);
                    candidate.alias_generation = image.alias_generation;
                    candidate.flags |= ImageFlagBits::GpuModified;
                    candidate.flags &= ~ImageFlagBits::Dirty;
                }
                SetAliasIdentity(state.backing, state.backing_uid, replacement,
                                 candidate.image_uid);
            } else {
                state.ResetAuthority();
            }
        } else if (!IsLiveImage(state.backing, state.backing_uid)) {
            state.ResetAuthority();
        }
        ASSERT(state.members > 1);
        if (--state.members == 1) {
            ForEachAlias(image, [](ImageId, Image& candidate) {
                candidate.flags &= ~ImageFlagBits::Aliased;
            });
            alias_states.erase(alias_it);
        }
    }

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        std::erase_if(pending_downloads, [image_id](const PendingImageDownload& d) {
            return d.image_id == image_id;
        });
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
