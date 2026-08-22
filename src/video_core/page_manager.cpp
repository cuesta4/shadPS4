// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <mutex>

#include <boost/container/small_vector.hpp>
#include <boost/icl/interval_set.hpp>
#include <tsl/robin_map.h>
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/performance_telemetry.h"
#include "common/range_lock.h"
#include "common/signal_context.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#ifdef ENABLE_USERFAULTFD
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include "common/error.h"
#endif
#else
#include <windows.h>
#include "common/spin_lock.h"
#endif

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

constexpr size_t PM_PAGE_SIZE = 4_KB;
constexpr size_t PM_PAGE_BITS = 12;

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers : 7;
        // At the moment only buffer cache can request read watchers.
        // And buffers cannot overlap, thus only 1 can exist per page.
        u8 num_read_watchers : 1;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <s32 delta, bool is_read>
        u8 AddDelta() {
            if constexpr (is_read) {
                if constexpr (delta == 1) {
                    return ++num_read_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                    return --num_read_watchers;
                } else {
                    return num_read_watchers;
                }
            } else {
                if constexpr (delta == 1) {
                    return ++num_write_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                    return --num_write_watchers;
                } else {
                    return num_write_watchers;
                }
            }
        }
    };

    struct WriteObserver {
        u64 id;
        MemoryWriteCallback callback;
        void* user_data;
    };

    struct WatchedPage {
        boost::container::small_vector<WriteObserver, 2> observers;
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / PAGES_PER_LOCK;
    inline static Vulkan::Rasterizer* rasterizer;
#ifdef ENABLE_USERFAULTFD
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
        uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        ASSERT_MSG(uffd != -1, "{}", Common::GetLastErrorMsg());

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        ASSERT(ret == 0 && api.api == UFFD_API);

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    void OnMap(VAddr address, size_t size) {
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed");
    }

    void OnUnmap(VAddr address, size_t size) {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed");
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? 0 : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            ASSERT_MSG(readret != -1 || errno == EAGAIN, "Unexpected result of uffd read");
            if (errno == EAGAIN) {
                continue;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            rasterizer->InvalidateMemory(addr, 1);
        }
    }

    std::jthread ufd_thread;
    int uffd;
#else
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void OnMap(VAddr address, size_t size) {
        // No-op
    }

    void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        impl.Protect(address, size, perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        if (Common::IsWriteError(context)) {
            return rasterizer->InvalidateMemory(addr, 8);
        } else {
            return rasterizer->ReadMemory(addr, 8);
        }
        return false;
    }
#endif

    template <bool track, bool is_read>
    void UpdatePageWatchers(VAddr addr, u64 size) {
        RENDERER_TRACE;

        size_t page = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        // Acquire locks for the range of pages
        const auto lock_start = locks.begin() + (page / PAGES_PER_LOCK);
        const auto lock_end = locks.begin() + Common::DivCeil(page_end, PAGES_PER_LOCK);
        Common::RangeLockGuard lk(lock_start, lock_end);

        auto perms = cached_pages[page].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // Perform pending (un)protect action
                Protect(range_begin << PM_PAGE_BITS, range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
        }

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];

            // Apply the change to the page state
            const u8 new_count = state.AddDelta<track ? 1 : -1, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // Only start a new range if the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current range up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    template <bool track, bool is_read>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) {
        RENDERER_TRACE;
        auto start_range = mask.FirstRange();
        auto end_range = mask.LastRange();

        if (start_range.second == end_range.second) {
            // if all pages are contiguous, use the regular UpdatePageWatchers
            const VAddr start_addr = base_addr + (start_range.first << PM_PAGE_BITS);
            const u64 size = (start_range.second - start_range.first) << PM_PAGE_BITS;
            return UpdatePageWatchers<track, is_read>(start_addr, size);
        }

        size_t base_page = (base_addr >> PM_PAGE_BITS);
        ASSERT(base_page % PAGES_PER_LOCK == 0);
        std::scoped_lock lk(locks[base_page / PAGES_PER_LOCK]);
        auto perms = cached_pages[base_page + start_range.first].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                // Perform pending (un)protect action
                Protect((range_begin << PM_PAGE_BITS), range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate pages
        for (size_t page = start_range.first; page < end_range.second; ++page) {
            PageState& state = cached_pages[base_page + page];
            const bool update = mask.Get(page);

            // Apply the change to the page state
            const u8 new_count =
                update ? state.AddDelta<track ? 1 : -1, is_read>() : state.AddDelta<0, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // If the page is not being updated, skip it
            if (!update) {
                continue;
            }

            // If the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current rango up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    void GpuMap(VAddr address, size_t size) {
        std::scoped_lock lock{mapping_mutex};
        OnMap(address, size);
        gpu_mappings += decltype(gpu_mappings)::interval_type::right_open(address, address + size);
        NotifyWrite(address, size, MemoryWriteSource::Map);
    }

    void GpuUnmap(VAddr address, size_t size) {
        std::scoped_lock lock{mapping_mutex};
        NotifyWrite(address, size, MemoryWriteSource::Unmap);
        OnUnmap(address, size);
        gpu_mappings -= decltype(gpu_mappings)::interval_type::right_open(address, address + size);
    }

    [[nodiscard]] MemoryWriteWatch ArmWriteWatch(VAddr address, MemoryWriteCallback callback,
                                                 void* user_data) {
        const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
        if (callback == nullptr) [[unlikely]] {
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::MemoryWatchArmFailures, 1);
            }
            return {};
        }

        const VAddr page = PageManager::GetPageAddr(address);
        std::scoped_lock mapping_lock{mapping_mutex};
        const auto page_range =
            decltype(gpu_mappings)::interval_type::right_open(page, page + PM_PAGE_SIZE);
        if (!boost::icl::contains(gpu_mappings, page_range)) [[unlikely]] {
            if (telemetry_enabled) {
                Common::PerformanceTelemetry::AddEnabled(
                    Common::PerformanceTelemetry::Counter::MemoryWatchArmFailures, 1);
            }
            return {};
        }

        std::scoped_lock lock{write_watch_mutex};
        auto [it, inserted] = watched_pages.try_emplace(page);
        if (inserted) {
            UpdatePageWatchers<true, false>(page, PM_PAGE_SIZE);
        }

        u64 id = next_watch_id++;
        if (id == 0) [[unlikely]] {
            id = next_watch_id++;
        }
        it.value().observers.push_back(WriteObserver{
            .id = id,
            .callback = callback,
            .user_data = user_data,
        });
        active_write_watches.fetch_add(1, std::memory_order_release);
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::MemoryWatchArms, 1);
        }
        return MemoryWriteWatch{
            .page = page,
            .id = id,
            .epoch = memory_epoch.load(std::memory_order_acquire),
        };
    }

    bool CancelWriteWatch(MemoryWriteWatch watch) {
        if (!watch) {
            return false;
        }

        std::scoped_lock lock{write_watch_mutex};
        const auto page_it = watched_pages.find(watch.page);
        if (page_it == watched_pages.end()) {
            return false;
        }
        auto& observers = page_it.value().observers;
        const auto observer_it = std::ranges::find(observers, watch.id, &WriteObserver::id);
        if (observer_it == observers.end()) {
            return false;
        }

        observers.erase(observer_it);
        active_write_watches.fetch_sub(1, std::memory_order_release);
        if (observers.empty()) {
            UpdatePageWatchers<false, false>(watch.page, PM_PAGE_SIZE);
            watched_pages.erase(page_it);
        }
        Common::PerformanceTelemetry::Add(
            Common::PerformanceTelemetry::Counter::MemoryWatchCancels);
        return true;
    }

    void NotifyWrite(VAddr address, u64 size, MemoryWriteSource source) {
        if (size == 0 || size - 1 > std::numeric_limits<VAddr>::max() - address) [[unlikely]] {
            return;
        }

        const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
        if (!telemetry_enabled &&
            active_write_watches.load(std::memory_order_acquire) == 0) [[likely]] {
            return;
        }
        NotifyWriteSlow(address, size, source);
    }

    SHAD_NO_INLINE void NotifyWriteSlow(VAddr address, u64 size, MemoryWriteSource source) {
        const bool telemetry_enabled = Common::PerformanceTelemetry::Enabled();
        Common::PerformanceTelemetry::ScopedDuration duration{
            telemetry_enabled, Common::PerformanceTelemetry::Counter::MemoryNotifyNs};
        const VAddr first_page = PageManager::GetPageAddr(address);
        const VAddr last_page = PageManager::GetPageAddr(address + size - 1);
        const u64 page_count = ((last_page - first_page) >> PM_PAGE_BITS) + 1;
        if (telemetry_enabled) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::MemoryNotifyCalls, 1);
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::MemoryNotifyPages, page_count);
            Common::PerformanceTelemetry::AddEnabled(NotifyCounter(source), 1);
        }

        if (active_write_watches.load(std::memory_order_acquire) == 0) [[likely]] {
            return;
        }
        const u64 epoch = memory_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;

        std::scoped_lock lock{write_watch_mutex};
        boost::container::small_vector<VAddr, 8> pages;
        if (page_count <= pages.capacity()) {
            for (VAddr page = first_page;; page += PM_PAGE_SIZE) {
                if (watched_pages.contains(page)) {
                    pages.push_back(page);
                }
                if (page == last_page) {
                    break;
                }
            }
        } else {
            pages.reserve(watched_pages.size());
            for (const auto& [page, watched] : watched_pages) {
                if (page >= first_page && page <= last_page) {
                    pages.push_back(page);
                }
            }
        }

        u64 callback_count{};
        for (const VAddr page : pages) {
            auto page_it = watched_pages.find(page);
            if (page_it == watched_pages.end()) {
                continue;
            }
            auto observers = std::move(page_it.value().observers);
            watched_pages.erase(page_it);
            active_write_watches.fetch_sub(observers.size(), std::memory_order_release);
            UpdatePageWatchers<false, false>(page, PM_PAGE_SIZE);
            callback_count += observers.size();
            for (const auto& observer : observers) {
                observer.callback(observer.user_data, page, epoch, source);
            }
        }

        if (telemetry_enabled && !pages.empty()) {
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::MemoryNotifyTrackedPages, pages.size());
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::MemoryNotifyCallbacks, callback_count);
            Common::PerformanceTelemetry::AddEnabled(
                Common::PerformanceTelemetry::Counter::MemoryWatchWakeups, callback_count);
        }
    }

    [[nodiscard]] static Common::PerformanceTelemetry::Counter NotifyCounter(
        MemoryWriteSource source) {
        using Counter = Common::PerformanceTelemetry::Counter;
        switch (source) {
        case MemoryWriteSource::Cpu:
            return Counter::MemoryNotifyCpu;
        case MemoryWriteSource::CommandProcessor:
            return Counter::MemoryNotifyCommandProcessor;
        case MemoryWriteSource::GpuCompletion:
            return Counter::MemoryNotifyGpuCompletion;
        case MemoryWriteSource::Map:
            return Counter::MemoryNotifyMap;
        case MemoryWriteSource::Unmap:
            return Counter::MemoryNotifyUnmap;
        }
        UNREACHABLE();
    }

    std::array<PageState, NUM_ADDRESS_PAGES> cached_pages{};
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
    using LockType = Common::AdaptiveMutex;
#else
    using LockType = Common::SpinLock;
#endif
    std::array<LockType, NUM_ADDRESS_LOCKS> locks{};
    std::mutex mapping_mutex;
    boost::icl::interval_set<VAddr> gpu_mappings;
    std::mutex write_watch_mutex;
    tsl::robin_map<VAddr, WatchedPage> watched_pages;
    std::atomic<u64> active_write_watches{};
    std::atomic<u64> memory_epoch{1};
    u64 next_watch_id{1};
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_)
    : impl{std::make_unique<Impl>(rasterizer_)} {}

PageManager::~PageManager() = default;

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->GpuMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->GpuUnmap(address, size);
}

MemoryWriteWatch PageManager::ArmWriteWatch(VAddr address, MemoryWriteCallback callback,
                                            void* user_data) {
    return impl->ArmWriteWatch(address, callback, user_data);
}

bool PageManager::CancelWriteWatch(MemoryWriteWatch watch) {
    return impl->CancelWriteWatch(watch);
}

void PageManager::NotifyWrite(VAddr address, u64 size, MemoryWriteSource source) {
    impl->NotifyWrite(address, size, source);
}

template <bool track>
void PageManager::UpdatePageWatchers(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<track, false>(addr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const {
    impl->UpdatePageWatchersForRegion<track, is_read>(base_addr, mask);
}

template void PageManager::UpdatePageWatchers<true>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchers<false>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchersForRegion<true, true>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<true, false>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, true>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, false>(VAddr base_addr,
                                                                     RegionBits& mask) const;

} // namespace VideoCore
