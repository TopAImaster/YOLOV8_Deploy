// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include<vector>
#include <algorithm>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static const int BatchSize = 1;
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int INPUT_C = 3;


static const int out_num = 8400 * 84;

const char* INPUT_NAME = "images";
const char* OUTPUT_NAME = "output0";

using namespace nvinfer1;
using namespace std;



class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;



void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);//engine.getNbBindings() 函数返回引擎对象 engine 中的绑定数量，即模型的输入和输出的总数。每个绑定对应一个模型的输入或输出。
    void* buffers[2];//这段代码定义了一个名为 buffers 的数组，其元素类型为 void*，即指针类型。

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));  //CHECK 核对校验  也可不使用
    CHECK(cudaMalloc(&buffers[outputIndex], out_num * sizeof(float)));//给cuda分配内存
    //cudaMalloc 函数来为 buffers[outputIndex] 分配一块内存空间。

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);//通常TensorRT的执行是异步的，因此将kernels加入队列放在CUDA stream流上
    cudaMemcpyAsync(output, buffers[outputIndex], out_num * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

ICudaEngine* inite_engine(std::string engine_path) {
    char* trtModelStream{ nullptr }; //指针函数,创建保存engine序列化文件结果
    size_t size{ 0 };
    // read model from the engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];//存储文件内容
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    // create a runtime (required for deserialization of model) with NVIDIA's logger
    IRuntime* runtime = createInferRuntime(gLogger); //反序列化方法
    assert(runtime != nullptr);
    // deserialize engine for using the char-stream
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    /*
    一个engine可以有多个execution context，并允许将同一套weights用于多个推理任务。
    可以在并行的CUDA streams流中按每个stream流一个engine和一个context来处理图像。
    每个context在engine相同的GPU上创建。
    */
    runtime->destroy();
    return engine;

};

cv::Mat ProcessImage(std::vector<cv::Mat> InputImage, float input_data[],cv::Mat M) {


    int ImgCount = InputImage.size();
    assert(ImgCount == BatchSize);
    //float input_data[BatchSize * 3 * INPUT_H * INPUT_W];

    cv::Mat originalImage = InputImage.at(0);
    cv::Mat img;


        // 进行仿射变换
    cv::Mat transformedImage;
    cv::warpAffine(originalImage, transformedImage, M, cv::Size(640, 640), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int w = transformedImage.cols;
    int h = transformedImage.rows;

    int i = 0;
    for (int row = 0; row < h; ++row) {
        uchar* uc_pixel = transformedImage.data + row * transformedImage.step;
        for (int col = 0; col < INPUT_W; ++col) {
            input_data[0 * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
            input_data[0 * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            input_data[0 * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    
    return transformedImage;
}



std::vector<std::string> readClassNames(const std::string& filename) {
    std::vector<std::string> classNames;

    std::ifstream file(filename);
    if (file.is_open()) {
        std::string className;
        while (std::getline(file, className)) {
            classNames.push_back(className);
        }
        file.close();
    }
    else {
        std::cout << "Failed to open file: " << filename << std::endl;
    }

    return classNames;
}



void video(IExecutionContext* context, std::vector<std::string> labels)
{
    
    cv::VideoCapture camera(1);
    if (!camera.isOpened()) {
        std::cerr << "Error: Failed to open the camera." << std::endl;
    }
    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    cv::Mat image;
    while (true) {
        // 读取摄像头视频的一帧
        camera.read(image);
        //

        int originalWidth = image.cols;
        int originalHeight = image.rows;

        // 目标图像大小
        int targetSize = 640;

        // 计算缩放比例
        float scale = static_cast<float>(targetSize) / std::max(originalWidth, originalHeight);

        // 计算缩放后的图像尺寸
        int targetWidth = static_cast<int>(originalWidth * scale);
        int targetHeight = static_cast<int>(originalHeight * scale);

        // 计算填充的宽度和高度
        int paddingWidth = targetSize - targetWidth;
        int paddingHeight = targetSize - targetHeight;

        // 计算左上角填充量
        int offsetX = paddingWidth / 2;
        int offsetY = paddingHeight / 2;

        // 定义变换矩阵
        cv::Mat M = cv::Mat::zeros(2, 3, CV_32FC1);

        // 设置缩放和平移参数
        M.at<float>(0, 0) = scale;
        M.at<float>(1, 1) = scale;
        M.at<float>(0, 2) = offsetX;
        M.at<float>(1, 2) = offsetY;


        vector<cv::Mat> InputImage;
        InputImage.push_back(image);




        float input[BatchSize * 3 * INPUT_H * INPUT_W];
        float output[out_num];






        cv::Mat transformedImage = ProcessImage(InputImage, input, M);




        doInference(*context, input, output, 1);



        //out(8400*84)

        cv::Mat dout(8400, 84, CV_32F, (float*)output);//将一串数组编程 mat类型的（8400*84）


        std::vector<cv::Rect> boxes;//std::vector<cv::Rect> 表示一个存储了 cv::Rect 对象的动态数组，其中 cv::Rect 是 OpenCV 库中的一个类，用于表示矩形区域。
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < dout.rows; i++) {
            cv::Mat classes_scores = dout.row(i).colRange(4, 84);//对 dout 的第 i 行进行操作，并提取了列索引从 4 到 84 的子区域。
            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

            // 置信度 0～1之间
            if (score > 0.25)
            {
                float cx = dout.at<float>(i, 0);
                float cy = dout.at<float>(i, 1);
                float ow = dout.at<float>(i, 2);
                float oh = dout.at<float>(i, 3);



                int x = static_cast<int>((cx - 0.5 * ow));//(ccx - 0.5 * ow)中心坐标转左上角坐标
                int y = static_cast<int>((cy - 0.5 * oh));
                int width = static_cast<int>(ow);
                int height = static_cast<int>(oh);




                cv::Rect box;
                box.x = x;
                box.y = y;
                box.width = width;
                box.height = height;

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }



        //NMS
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
        for (size_t i = 0; i < indexes.size(); i++) {
            int index = indexes[i];
            int idx = classIds[index];

            cv::rectangle(transformedImage, boxes[index], cv::Scalar(0, 0, 255), 2, 8);//画的框

            cv::rectangle(transformedImage, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);//画的类别背景

            putText(transformedImage, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);//写的类别

        }


        cv::Mat inverseM;
        cv::invertAffineTransform(M, inverseM);
        cv::warpAffine(transformedImage, image, inverseM, cv::Size(originalWidth, originalHeight), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));



        //cv::imshow("YOLOv8+ONNXRUNTIME 对象检测演示", image);
        //cv::waitKey(0);

        //cv::Mat inverseM;

        //cv::invertAffineTransform(M, inverseM);

        //cv::warpAffine(transformedImage, image, inverseM, cv::Size(originalWidth, originalHeight), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        //// 显示结果
        //cv::imshow("Original Image", image);
        //cv::waitKey(0);

        //std::cout << "end" << std::endl;

        //
        //cv::putText(image, " ", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        // 在窗口上显示摄像头视频的帧
        cv::imshow("Camera", image);

        // 按下 ESC 键退出循环
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
}


void _serialize_engine(ICudaEngine* engine)
{
    IHostMemory* modelStream{ nullptr };
    modelStream = engine->serialize();
    ofstream f("yolo_8.engine", ios::binary);
    // f << modelStream->data();
    f.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    f.close();
    modelStream->destroy();
}

// 从文件中读取序列化后的引擎并反序列化
ICudaEngine* _deserialize_engine()
{
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = nullptr;

    // 读取文件
    ifstream file("yolo_8.engine", std::ios::binary);
    if (file.good()) {
        // 获取文件大小
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        // 分配内存
        vector<char> trtModelStream(size);
        assert(trtModelStream.data());

        // 读取文件内容
        file.read(trtModelStream.data(), size);
        file.close();

        // 反序列化引擎
        engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
    }

    // 销毁不需要的资源
    runtime->destroy();

    // 返回引擎
    return engine;
}


void build_engine() {
    // 创建构建器
    IBuilder* builder = createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    IBuilderConfig* config = builder->createBuilderConfig();

    // 创建网络模型
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    //INetworkDefinition* network = builder->createNetworkV2();

    // 解析ONNX模型
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    bool parser_status = parser->parseFromFile("yolov8m_smi3.onnx", static_cast<int>(ILogger::Severity::kWARNING));

    // 构建引擎
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 30);  // 1GB
    ICudaEngine* engine1 = builder->buildEngineWithConfig(*network, *config);
    _serialize_engine(engine1);

    // 反序列化引擎并创建执行上下文
    ICudaEngine* engine2 = _deserialize_engine();
    IExecutionContext* context = engine2->createExecutionContext();

    // 销毁不需要的资源
    context->destroy();
    engine2->destroy();
    engine1->destroy();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

   // std::cout << "按下任意键退出程序..." << std::endl;

}


bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}


int main() {

    std::string filename = "yolo_8.engine";

    if (!fileExists(filename)) {
        build_engine();
    }

    std::vector<std::string> labels = readClassNames("coco_class_names.txt");
    //加载并初始化TRT引擎
    std::string enginePath = "yolo_8.engine";
    ICudaEngine* engine = inite_engine(enginePath);
    IExecutionContext* context = engine->createExecutionContext();
    video(context, labels);


    //cv::Mat image = cv::imread("F:\\c++\\yolov8\\bus.jpg");
    //int iw = image.cols;
    //int ih = image.rows;
    //float w_sclae = static_cast<float>(INPUT_W) / static_cast<float>(iw);
    //float h_sclae = static_cast<float>(INPUT_H) / static_cast<float>(ih);

    

    return 0;

}
