#include <cupti.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr size_t kActivityBufferSize = 16 * 1024 * 1024;
constexpr size_t kActivityBufferAlign = 8;

struct KernelActivityRecord {
    std::string name;
    uint32_t device_id;
    uint32_t stream_id;
    uint32_t correlation_id;
    uint64_t start_ns;
    uint64_t end_ns;
};

std::mutex g_records_mutex;
bool g_callbacks_registered = false;
std::atomic<bool> g_capture_active = false;
std::vector<KernelActivityRecord> g_records;
std::string g_async_error;
CUpti_ActivityKind g_enabled_activity_kind = CUPTI_ACTIVITY_KIND_INVALID;

std::string cupti_error_string(CUptiResult result) {
    const char* error_string = nullptr;
    cuptiGetResultString(result, &error_string);
    return error_string == nullptr ? "unknown CUPTI error" : error_string;
}

void throw_on_cupti_error(CUptiResult result, const char* expr) {
    if (result == CUPTI_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(expr) + ": " + cupti_error_string(result));
}

#define CUPTI_CHECK(expr) throw_on_cupti_error((expr), #expr)

void record_kernel_activity(const CUpti_ActivityKernel9& kernel) {
    if (!g_capture_active) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_records_mutex);
    g_records.push_back(KernelActivityRecord{
        .name = kernel.name == nullptr ? "" : kernel.name,
        .device_id = kernel.deviceId,
        .stream_id = kernel.streamId,
        .correlation_id = kernel.correlationId,
        .start_ns = kernel.start,
        .end_ns = kernel.end,
    });
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
    void* raw_ptr = nullptr;
    if (posix_memalign(&raw_ptr, kActivityBufferAlign, kActivityBufferSize) != 0) {
        *buffer = nullptr;
        *size = 0;
        *max_num_records = 0;
        return;
    }
    *buffer = static_cast<uint8_t*>(raw_ptr);
    *size = kActivityBufferSize;
    *max_num_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t size, size_t valid_size) {
    if (buffer == nullptr || size == 0) {
        return;
    }

    CUptiResult status = CUPTI_SUCCESS;
    CUpti_Activity* record = nullptr;
    while ((status = cuptiActivityGetNextRecord(buffer, valid_size, &record)) == CUPTI_SUCCESS) {
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                record_kernel_activity(*reinterpret_cast<const CUpti_ActivityKernel9*>(record));
                break;
            default:
                break;
        }
    }

    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
        std::lock_guard<std::mutex> lock(g_records_mutex);
        g_async_error = std::string("cuptiActivityGetNextRecord: ") + cupti_error_string(status);
        free(buffer);
        return;
    }

    size_t dropped = 0;
    CUptiResult dropped_status = cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    free(buffer);
    if (dropped_status != CUPTI_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_records_mutex);
        g_async_error =
            std::string("cuptiActivityGetNumDroppedRecords: ") + cupti_error_string(dropped_status);
        return;
    }
    if (dropped != 0) {
        std::lock_guard<std::mutex> lock(g_records_mutex);
        g_async_error = "CUPTI dropped activity records";
    }
}

void ensure_callbacks_registered() {
    if (g_callbacks_registered) {
        return;
    }
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
    g_callbacks_registered = true;
}

class CuptiKernelActivityProfiler {
public:
    CuptiKernelActivityProfiler() {
        ensure_callbacks_registered();
    }

    void start() {
        std::lock_guard<std::mutex> lock(g_records_mutex);
        if (g_capture_active) {
            throw std::runtime_error("CUPTI capture already active");
        }
        g_records.clear();
        g_async_error.clear();
        g_capture_active = true;
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        g_enabled_activity_kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    }

    py::list stop() {
        if (!g_capture_active) {
            throw std::runtime_error("CUPTI capture is not active");
        }

        CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        if (g_enabled_activity_kind != CUPTI_ACTIVITY_KIND_INVALID) {
            CUPTI_CHECK(cuptiActivityDisable(g_enabled_activity_kind));
            g_enabled_activity_kind = CUPTI_ACTIVITY_KIND_INVALID;
        }

        std::vector<KernelActivityRecord> records;
        {
            std::lock_guard<std::mutex> lock(g_records_mutex);
            if (!g_async_error.empty()) {
                const auto error = g_async_error;
                g_async_error.clear();
                g_records.clear();
                g_capture_active = false;
                throw std::runtime_error(error);
            }
            records = g_records;
            g_records.clear();
            g_capture_active = false;
        }

        py::list out;
        for (const auto& record : records) {
            py::dict entry;
            entry["name"] = record.name;
            entry["device_id"] = record.device_id;
            entry["stream_id"] = record.stream_id;
            entry["correlation_id"] = record.correlation_id;
            entry["start_ns"] = record.start_ns;
            entry["end_ns"] = record.end_ns;
            entry["duration_ns"] = record.end_ns - record.start_ns;
            out.append(std::move(entry));
        }
        return out;
    }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CuptiKernelActivityProfiler>(m, "CuptiKernelActivityProfiler")
        .def(py::init<>())
        .def("start", &CuptiKernelActivityProfiler::start)
        .def("stop", &CuptiKernelActivityProfiler::stop);
}
