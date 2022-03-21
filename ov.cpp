#include <opencv2/opencv.hpp>
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
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        // cv::imshow("", cv::Mat(cv::Size{1280, 720}, CV_8UC3, cv::Scalar{0, 0, 0}));
        cv::imshow("", cv::Mat(cv::Size{300, 300}, CV_8UC3, cv::Scalar{0, 0, 0}));
        cv::waitKey(1);
    }
    auto end = chrono::steady_clock::now();
    cout << double((end - start).count()) / 1000 / 1000 << '\n';
    return 0;
}
