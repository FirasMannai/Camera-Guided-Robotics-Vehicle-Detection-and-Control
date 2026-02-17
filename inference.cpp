#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "hsh_inference/msg/model_output.hpp"

using namespace nvinfer1;
using namespace std::placeholders;

// =======================================================
// CUDA CHECK
// =======================================================
#define CHECK(status)                                     \
    do {                                                  \
        auto ret = (status);                              \
        if (ret != cudaSuccess) {                         \
            std::cerr << "CUDA Error: " << ret << "\n";   \
            std::abort();                                 \
        }                                                 \
    } while (0)

// =======================================================
// TensorRT Logger
// =======================================================
class TRTLogger : public ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

// =======================================================
// Options struct
// =======================================================
struct Options {
    std::string engine_file = "";
    std::string engine_path = "";
};

// =======================================================
// Inference Node
// =======================================================
class InferenceNode : public rclcpp::Node
{
public:
    InferenceNode() : Node("inference")
    {
        // -------------------------------
        // Load parameters into opt
        // -------------------------------
        this->declare_parameter<std::string>("engine_file", "");
        this->declare_parameter<std::string>("engine_path", "");
         this->declare_parameter<std::string>("model_type", "yolo"); // default: yolo

        Options opt;
        this->get_parameter("engine_file", opt.engine_file);
        this->get_parameter("engine_path", opt.engine_path);
        
        this->get_parameter("model_type", model_type_);

        engine_path_ = opt.engine_path + "/" + opt.engine_file;
        RCLCPP_INFO(this->get_logger(), "Loading TRT engine: %s",
                    engine_path_.c_str());

        // -------------------------------
        // Load engine file
        // -------------------------------
        std::vector<char> engine_data = load_engine_file(engine_path_);
        if (engine_data.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Engine file NOT FOUND!");
            rclcpp::shutdown();
            return;
        }

        // -------------------------------
        // TensorRT runtime / engine / context
        // -------------------------------
        logger_ = new TRTLogger();
        runtime_ = createInferRuntime(*logger_);
        engine_  = runtime_->deserializeCudaEngine(engine_data.data(),
                                                   engine_data.size());
        context_ = engine_->createExecutionContext();

        if (!engine_ || !context_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create TRT engine!");
            rclcpp::shutdown();
            return;
        }

        // -------------------------------
        // Automatically find input/output bindings
        // -------------------------------
        inputIndex_ = -1;
        outputIndex_ = -1;
        std::string in_dims_str = "[";
        std::string out_dims_str = "]";
        

        // Store input dims for resizing
        INPUT_C = INPUT_H = INPUT_W = 0;

        for (int i = 0; i < engine_->getNbBindings(); ++i)
        {
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);

            if (engine_->bindingIsInput(i)) {
                inputIndex_ = i;
                inputName_ = engine_->getBindingName(i);

                in_dims_str.clear();
                in_dims_str += "[";
                dim_in_.clear();
                for (int d = 0; d < dims.nbDims; ++d) {
                    in_dims_str += std::to_string(dims.d[d]);
                    if (d < dims.nbDims - 1) in_dims_str += " x ";
                    dim_in_.push_back(dims.d[d]);


                }
                in_dims_str += "]";

                // Dynamically set INPUT_C/H/W for preprocessing
                if (dims.nbDims == 4) {  // NCHW
                    INPUT_C = dims.d[1];
                    INPUT_H = dims.d[2];
                    INPUT_W = dims.d[3];
                } else if (dims.nbDims == 3) {  // CHW
                    INPUT_C = dims.d[0];
                    INPUT_H = dims.d[1];
                    INPUT_W = dims.d[2];
                } else {
                    RCLCPP_WARN(this->get_logger(),
                                "Unexpected input dims, resizing may fail!");
                }

            } else { // output binding
                outputIndex_ = i;
                outputName_ = engine_->getBindingName(i);
                
                outputSize_ = 1;
                out_dims_str.clear();
                dim_out_.clear();
                out_dims_str += "[";
                for (int d = 0; d < dims.nbDims; ++d) {
                    outputSize_ *= dims.d[d];
                    out_dims_str += std::to_string(dims.d[d]);
                    if (d < dims.nbDims - 1) out_dims_str += " x ";
                    dim_out_.push_back(dims.d[d]);
                }
                out_dims_str += "]";
            }
        }

RCLCPP_INFO(this->get_logger(),
            "Input: %s, Output: %s, Input dims: %s, Output dims: %s",
            inputName_.c_str(), outputName_.c_str(),
            in_dims_str.c_str(), out_dims_str.c_str());

        if (inputIndex_ < 0 || outputIndex_ < 0) {
            RCLCPP_ERROR(this->get_logger(),
                         "No input or output binding found!");
            rclcpp::shutdown();
            return;
        }

        // RCLCPP_INFO(this->get_logger(),
        //             "Input: %s, Output: %s, Input dims: %dx%dx%d",
        //             inputName_.c_str(), outputName_.c_str(),
        //             INPUT_C, INPUT_H, INPUT_W);

        // -------------------------------
        // Allocate buffers
        // -------------------------------
        inputCPU_.resize(INPUT_C * INPUT_H * INPUT_W);
        outputCPU_.resize(outputSize_);

        CHECK(cudaMalloc(&buffers_[inputIndex_],
                         inputCPU_.size() * sizeof(float)));
        CHECK(cudaMalloc(&buffers_[outputIndex_],
                         outputSize_ * sizeof(float)));

        CHECK(cudaStreamCreate(&stream_));

        // -------------------------------
        // ROS interfaces
        // -------------------------------
        std::string topic_name = (model_type_ == "yolo") ?
                                 "hsh_camera/img" : "hsh_camera/img_224x224";

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic_name, 1,
            std::bind(&InferenceNode::callback, this, _1));

        pub_ = this->create_publisher<hsh_inference::msg::ModelOutput>(
            "/hsh_inference/model_output", 1);
    }

    ~InferenceNode()
    {
        cudaStreamDestroy(stream_);
        cudaFree(buffers_[inputIndex_]);
        cudaFree(buffers_[outputIndex_]);

        delete context_;
        delete engine_;
        delete runtime_;
        delete logger_;
    }

private:

    // =======================================================
    // Load engine file into vector<char>
    // =======================================================
    std::vector<char> load_engine_file(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.good())
            return {};

        f.seekg(0, std::ifstream::end);
        size_t size = f.tellg();
        f.seekg(0, std::ifstream::beg);

        std::vector<char> data(size);
        f.read(data.data(), size);
        return data;
    }

    // =======================================================
    // Image preprocessing → CHW float32
    // =======================================================
    void preprocess(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat img = cv_bridge::toCvCopy(msg, "bgr8")->image;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

        if (model_type_ == "resnet"){
             // Define mean and std for resnet (same values used in training) (see /tools/deep_learning_training/lane_and_objbox/train_lane-6pt_and_objbox.py)
            std::vector<float> mean = {0.485f, 0.456f, 0.406f}; 
            std::vector<float> std = {0.229f, 0.224f, 0.225f};

            // Apply normalization (ImageNet normalization: (x - mean) / std)
            for (int i = 0; i < 3; i++) {
                resized.forEach<cv::Vec3f>([&](cv::Vec3f &pixel, const int *position) -> void {
                    pixel[i] = (pixel[i] - mean[i]) / std[i];  // Normalize each channel (R, G, B)
                });
            }

        } 

       

        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);

        int area = INPUT_W * INPUT_H;
        memcpy(inputCPU_.data(),
               channels[0].data, area * sizeof(float));
        memcpy(inputCPU_.data() + area,
               channels[1].data, area * sizeof(float));
        memcpy(inputCPU_.data() + 2 * area,
               channels[2].data, area * sizeof(float));
    }

    // =======================================================
    // Image callback
    // =======================================================
    void callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        preprocess(msg);

        CHECK(cudaMemcpyAsync(buffers_[inputIndex_], inputCPU_.data(),
                              inputCPU_.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_));

        context_->enqueueV2(buffers_, stream_, nullptr);

        CHECK(cudaMemcpyAsync(outputCPU_.data(), buffers_[outputIndex_],
                              outputSize_ * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));

        cudaStreamSynchronize(stream_);

        hsh_inference::msg::ModelOutput out;
        out.header = msg->header;
        out.dim_in  = dim_in_;
        out.dim_out = dim_out_;

        // Fill output with EXACT model output size
        out.data.assign(outputCPU_.begin(), outputCPU_.end());

        pub_->publish(out);
    }

    // =======================================================
    // Internal fields
    // =======================================================
    int INPUT_C, INPUT_H, INPUT_W;

    std::string engine_path_;
    std::string model_type_;
    std::string inputName_, outputName_;

    TRTLogger *logger_;
    IRuntime *runtime_;
    ICudaEngine *engine_;
    IExecutionContext *context_;

    int inputIndex_;
    int outputIndex_;

    std::vector<float> inputCPU_;
    std::vector<float> outputCPU_;

    int outputSize_;

    void *buffers_[2];
    cudaStream_t stream_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<hsh_inference::msg::ModelOutput>::SharedPtr pub_;

    std::vector<int32_t> dim_in_;
    std::vector<int32_t> dim_out_;
};

// =======================================================
// MAIN
// =======================================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<InferenceNode>());
    rclcpp::shutdown();
    return 0;
}
