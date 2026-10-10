#include <thread>
#include <vector>

#include "../tent_native_trace.cpp"

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    const std::string mode(argv[1]);
    if (mode == "complete") {
        const char* stages[] = {"metadata_rpc", "endpoint_construct", "endpoint_connect",
                                "bootstrap_rpc"};
        std::vector<std::thread> workers;
        for (size_t worker = 0; worker < 8; ++worker) {
            workers.emplace_back([stage = stages[worker % 4]] {
                for (size_t i = 0; i < 128; ++i) {
                    Span span(stage);
                    span.finish(0);
                }
            });
        }
        for (auto& worker : workers) worker.join();
        return 0;
    }
    if (mode == "active") {
        std::atomic<bool> started{false};
        std::thread worker([&started] {
            Span span("endpoint_construct");
            started.store(true, std::memory_order_release);
            while (true) std::this_thread::yield();
        });
        while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
        worker.detach();
        return 0;
    }
    if (mode == "unpublished") {
        count.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    return 1;
}
