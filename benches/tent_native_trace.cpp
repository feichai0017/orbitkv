// Benchmark-only interposition for Mooncake 71973589 on Linux x86-64/libstdc++.
// No transport setting, descriptor, return value or ownership is changed.
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "tent/common/status.h"

namespace mooncake::tent {
struct BootstrapDesc;
class RdmaContext;
struct EndPointParams;
}  // namespace mooncake::tent

using mooncake::tent::BootstrapDesc;
using mooncake::tent::EndPointParams;
using mooncake::tent::RdmaContext;
using mooncake::tent::Status;

namespace {
constexpr size_t kLimit = 4096;
struct Event {
    const char* stage;
    uint64_t begin_ns;
    uint64_t end_ns;
    long tid;
    int status;
};
Event events[kLimit];
std::atomic<size_t> count{0};

uint64_t now() {
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000 + value.tv_nsec;
}

template <typename Function>
Function original(const char* symbol) {
    auto address = dlsym(RTLD_NEXT, symbol);
    if (!address) {
        dprintf(STDERR_FILENO, "TENT trace missing native symbol: %s\n", symbol);
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

class Span {
   public:
    explicit Span(const char* stage) : stage_(stage), begin_(now()), tid_(syscall(SYS_gettid)) {}
    ~Span() {
        const uint64_t end = now();
        const size_t index = count.fetch_add(1, std::memory_order_relaxed);
        if (index < kLimit) events[index] = Event{stage_, begin_, end, tid_, status_};
    }
    void finish(int status) { status_ = status; }

   private:
    const char* stage_;
    uint64_t begin_;
    long tid_;
    int status_ = -1;
};

__attribute__((destructor)) void write_events() {
    const char* path = std::getenv("TENT_NATIVE_TRACE_PATH");
    if (!path) return;
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        perror("TENT trace output");
        _exit(2);
    }
    FILE* output = fdopen(fd, "w");
    if (!output) {
        close(fd);
        _exit(2);
    }
    const size_t seen = count.load(std::memory_order_acquire);
    const size_t retained = seen < kLimit ? seen : kLimit;
    std::fprintf(output,
                 "{\"schema\":\"tent.native-trace.v1\",\"pid\":%ld,"
                 "\"clock\":\"CLOCK_MONOTONIC\",\"seen\":%zu,"
                 "\"limit\":%zu,\"overflow\":%s,\"events\":[",
                 static_cast<long>(getpid()), seen, kLimit, seen > kLimit ? "true" : "false");
    for (size_t i = 0; i < retained; ++i) {
        const auto& event = events[i];
        std::fprintf(output,
                     "%s{\"stage\":\"%s\",\"begin_mono_ns\":%llu,"
                     "\"end_mono_ns\":%llu,\"tid\":%ld,\"status\":%d}",
                     i ? "," : "", event.stage, static_cast<unsigned long long>(event.begin_ns),
                     static_cast<unsigned long long>(event.end_ns), event.tid, event.status);
    }
    std::fprintf(output, "]}\n");
    const bool failed = std::ferror(output);
    if (std::fclose(output) != 0 || failed) _exit(2);
}
}  // namespace

// These symbol names and nontrivial Status return ABI are pinned, not a public
// Mooncake API. Use its exact status header and check exports before deployment.
extern "C" Status trace_metadata(const std::string&, std::string&) asm(
    "_ZN8mooncake4tent13ControlClient14getSegmentDescERKNSt7__cxx1112basic_stringIcSt11char_"
    "traitsIcESaIcEEERS7_");
extern "C" Status trace_metadata(const std::string& address, std::string& response) {
    using Function = Status (*)(const std::string&, std::string&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent13ControlClient14getSegmentDescERKNSt7__cxx1112basic_stringIcSt11char_"
        "traitsIcESaIcEEERS7_");
    Span span("metadata_rpc");
    auto status = function(address, response);
    span.finish(static_cast<int>(status.code()));
    return status;
}

extern "C" Status trace_bootstrap(const std::string&, const BootstrapDesc&, BootstrapDesc&) asm(
    "_ZN8mooncake4tent13ControlClient9bootstrapERKNSt7__cxx1112basic_stringIcSt11char_"
    "traitsIcESaIcEEERKNS0_13BootstrapDescERSA_");
extern "C" Status trace_bootstrap(const std::string& address, const BootstrapDesc& request,
                                  BootstrapDesc& response) {
    using Function = Status (*)(const std::string&, const BootstrapDesc&, BootstrapDesc&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent13ControlClient9bootstrapERKNSt7__cxx1112basic_stringIcSt11char_"
        "traitsIcESaIcEEERKNS0_13BootstrapDescERSA_");
    Span span("bootstrap_rpc");
    auto status = function(address, request, response);
    span.finish(static_cast<int>(status.code()));
    return status;
}

extern "C" int trace_construct(void*, RdmaContext*, EndPointParams*, const std::string&) asm(
    "_ZN8mooncake4tent12RdmaEndPoint9constructEPNS0_11RdmaContextEPNS0_14EndPointParamsERKNSt7__"
    "cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
extern "C" int trace_construct(void* self, RdmaContext* context, EndPointParams* params,
                               const std::string& name) {
    using Function = int (*)(void*, RdmaContext*, EndPointParams*, const std::string&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent12RdmaEndPoint9constructEPNS0_11RdmaContextEPNS0_14EndPointParamsERKNSt7_"
        "_cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    Span span("endpoint_construct");
    const int status = function(self, context, params, name);
    span.finish(status);
    return status;
}

extern "C" Status
trace_connect(void*, const std::string&, const std::string&, const std::string&) asm(
    "_ZN8mooncake4tent12RdmaEndPoint7connectERKNSt7__cxx1112basic_stringIcSt11char_"
    "traitsIcESaIcEEES9_S9_");
extern "C" Status trace_connect(void* self, const std::string& server, const std::string& nic,
                                const std::string& rpc) {
    using Function = Status (*)(void*, const std::string&, const std::string&, const std::string&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent12RdmaEndPoint7connectERKNSt7__cxx1112basic_stringIcSt11char_"
        "traitsIcESaIcEEES9_S9_");
    Span span("endpoint_connect");
    auto status = function(self, server, nic, rpc);
    span.finish(static_cast<int>(status.code()));
    return status;
}
