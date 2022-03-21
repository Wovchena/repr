#include <openvino/openvino.hpp>
using namespace ov;
using namespace std;

static inline void catcher() noexcept {
    if (std::current_exception()) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& error) {
            cerr << error.what() << endl;
        } catch (...) {
            cerr << "Non-exception object thrown" << endl;
        }
        std::exit(1);
    }
    std::abort();
}

int main() {
    set_terminate(catcher);
    {
        InferRequest ireq = 
            Core{}.compile_model("C:\\Users\\vzlobin\\Downloads\\d\\intel\\face-detection-retail-0004\\FP32\\face-detection-retail-0004.xml")
            .create_infer_request();
        auto start = chrono::steady_clock::now();
        int nruns = 1000;
        for (int i = 0; i < nruns; ++i) {
            ireq.start_async();
            ireq.wait_for(chrono::milliseconds{0});
            ireq.wait();
        }
        auto end = chrono::steady_clock::now();
        cout << double((end - start).count()) / nruns / 1000 << '\n';
    }

    {
        InferRequest ireq = 
            Core{}.compile_model("C:\\Users\\vzlobin\\Downloads\\d\\intel\\face-detection-retail-0004\\FP32\\face-detection-retail-0004.xml")
            .create_infer_request();
        mutex mtx;
        condition_variable cv;
        bool finished = false;
        auto start = chrono::steady_clock::now();
        int nruns = 1000;
            ireq.set_callback([&](exception_ptr) {
                {
                    lock_guard<std::mutex> lock(mtx);
                    finished = true;
                }
                cv.notify_one();
            });
        for (int i = 0; i < nruns; ++i) {
            ireq.start_async();
            unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]{return finished;});
            finished = false;
        }
        auto end = chrono::steady_clock::now();
        cout << double((end - start).count()) / nruns / 1000 << '\n';
    }
    return 0;
}
