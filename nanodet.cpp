#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
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

// stuff we know about the network and the input/output blobs
#define BATCH_SIZE 1
static const int INPUT_H = 416;
static const int INPUT_W = 416;
static const int REG_MAX = 7;
static const int CLASS_NUM = 80;
static const int OUTPUT_SIZE = 1000 * 7;//tmp

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* upsample(INetworkDefinition *network, ITensor& input, int res_w, int res_h, int ch2)
{
    auto upsample1 = network->addResize(input);
    assert(upsample1);
    upsample1->setResizeMode(ResizeMode::kLINEAR);
    upsample1->setOutputDimensions(Dims3{ch2, res_h, res_w});
    return upsample1;
}

ILayer* shortcut(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
                   int inch, int outch, int dw_kernel_size, int groups)
{
    int padding = int((dw_kernel_size -1 ) / 2);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* shortcut_conv1 = network->addConvolutionNd(input, inch, DimsHW{dw_kernel_size, dw_kernel_size}, weightMap[lname + "0.weight"], emptywts);
    assert(shortcut_conv1);
    shortcut_conv1->setStrideNd(DimsHW{1, 1});
    shortcut_conv1->setPaddingNd(DimsHW{padding, padding});
    shortcut_conv1->setNbGroups(groups);
    IScaleLayer *bn5 = addBatchNorm2d(network, weightMap, *shortcut_conv1->getOutput(0), lname + "1", 1e-5);
    IActivationLayer* relu5 = network->addActivation(*bn5->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu5);

    IConvolutionLayer* shortcut_conv2 = network->addConvolutionNd(*relu5->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "2.weight"], emptywts);
    assert(shortcut_conv2);
    shortcut_conv2->setStrideNd(DimsHW{1, 1});
    IScaleLayer *bn6 = addBatchNorm2d(network, weightMap, *shortcut_conv2->getOutput(0), lname + "3", 1e-5);
    IActivationLayer* relu6 = network->addActivation(*bn6->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu6);
    return relu6;
}                   

ILayer* GhostModule(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
                   int inch, int outch)
{
    int ratio = 2;
    int init_channels = int(outch / ratio);
    int new_channels = init_channels * (ratio - 1);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* primary_conv = network->addConvolutionNd(input, init_channels, DimsHW{1, 1}, weightMap[lname + "primary_conv.0.weight"], emptywts);
    assert(primary_conv);
    primary_conv->setStrideNd(DimsHW{1, 1});
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *primary_conv->getOutput(0), lname + "primary_conv.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu1);

    IConvolutionLayer* cheap_operation = network->addConvolutionNd(*relu1->getOutput(0), new_channels, DimsHW{3, 3}, weightMap[lname + "cheap_operation.0.weight"], emptywts);
    assert(cheap_operation);
    cheap_operation->setStrideNd(DimsHW{1, 1});
    cheap_operation->setPaddingNd(DimsHW{1, 1});
    cheap_operation->setNbGroups(init_channels);
    IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *cheap_operation->getOutput(0), lname + "cheap_operation.1", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu2);
    ITensor* inputTensors[] = { relu1->getOutput(0), relu2->getOutput(0)};
    auto Ghost_out = network->addConcatenation(inputTensors, 2);
    return Ghost_out;
}
ILayer* GhostBottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
                    int inch, int midch, int outch, int kernel_size)
{
    auto ghost1 = GhostModule(network, weightMap, input, lname + "ghost1.", inch, midch);

    auto ghost2 = GhostModule(network, weightMap, *ghost1->getOutput(0), lname + "ghost2.", midch, outch);
 
    auto shcut_out = shortcut(network, weightMap, input, lname + "shortcut.", inch, outch, kernel_size, inch);
     //ITensor* inputTensors4[] = { Ghost2_out->getOutput(0), relu6->getOutput(0)};
    auto GhostBot_out = network->addElementWise(*ghost2->getOutput(0),*shcut_out->getOutput(0), ElementWiseOperation::kSUM);
    return GhostBot_out;
}

ILayer* GhostBlocks(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
                    int inch, int outch, int expand, int kernel_size, int num_blocks)
{
    ILayer* Ghost_out  = GhostBottleneck(network, weightMap, input, lname + "blocks.0.", inch, int(outch * expand), outch,  kernel_size);;
    for (int i = 0; i < num_blocks - 1; i++)
    {
        Ghost_out = GhostBottleneck(network, weightMap, *Ghost_out->getOutput(0), lname, inch, int(outch * expand), outch, kernel_size);
    }
    return Ghost_out;
}

ILayer* DepthwiseConvModule(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, 
                    int inch, int outch, int kernel_size, int stride)
{
    int padding = int(kernel_size/2);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* depthwise1 = network->addConvolutionNd(input, inch, DimsHW{kernel_size, kernel_size}, weightMap[lname + "depthwise.weight"], emptywts);
    assert(depthwise1);
    depthwise1->setStrideNd(DimsHW{stride, stride});
    depthwise1->setPaddingNd(DimsHW{padding, padding});
    depthwise1->setNbGroups(inch);
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *depthwise1->getOutput(0), lname + "dwnorm", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu1);

    IConvolutionLayer* pointwise1 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "pointwise.weight"], emptywts);
    assert(pointwise1);
    pointwise1->setStrideNd(DimsHW{1, 1});
    IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *pointwise1->getOutput(0), lname + "pwnorm", 1e-5);
    auto relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu2);
    return relu2;
}

ILayer* plus_head(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& in_tensor, 
                std::string lname, int outch, int index)
{
    auto feat = DepthwiseConvModule(network, weightMap, in_tensor, lname + "cls_convs." + std::to_string(index) + ".0.", outch, outch, 5,1);
    feat = DepthwiseConvModule(network, weightMap, *feat->getOutput(0), lname + "cls_convs." + std::to_string(index) + ".1.", outch, outch, 5, 1);
    IConvolutionLayer* reg_conv = network->addConvolutionNd(*feat->getOutput(0), CLASS_NUM + 4 * (REG_MAX + 1), DimsHW{1, 1}, weightMap[lname + "gfl_cls." + std::to_string(index) +".weight"], weightMap[lname + "gfl_cls." + std::to_string(index) +".bias"]);
    assert(reg_conv);
    reg_conv->setStrideNd(DimsHW{1, 1});

    Dims dims = reg_conv->getOutput(0)->getDimensions();
    auto out = network->addShuffle(*reg_conv->getOutput(0));
    assert(out);
    out->setReshapeDimensions(Dims2(dims.d[0], dims.d[1]*dims.d[2]));
    out->setSecondTranspose(Permutation{1, 0});
    return out;
}


ILayer* extra_lvl(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& in_tensor, ITensor& out_tensor, 
                std::string lname, int outch)
{

    auto extra_lvl_in = DepthwiseConvModule(network, weightMap, in_tensor, lname + "extra_lvl_in_conv.0.", outch, outch, 5, 2);
    auto extra_lvl_out = DepthwiseConvModule(network, weightMap, out_tensor, lname + "extra_lvl_out_conv.0.", outch, outch, 5, 2);
     //ITensor* inputTensors4[] = { Ghost2_out->getOutput(0), relu6->getOutput(0)};
    auto out = network->addElementWise(*extra_lvl_in->getOutput(0),*extra_lvl_out->getOutput(0), ElementWiseOperation::kSUM);
    return out;
}

ILayer* bottom_up(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& feat_heigh, ITensor& feat_low, 
                std::string lname, int outch, int index)
{

    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    auto downsample_feat = DepthwiseConvModule(network, weightMap, feat_low, lname + "downsamples." + std::to_string(index) + ".", outch, outch, 5, 2);

    ITensor* inputTensors[] = { downsample_feat->getOutput(0), &feat_heigh};
    auto cat_out = network->addConcatenation(inputTensors, 2);
    auto inner_out = GhostBlocks(network, weightMap, *cat_out->getOutput(0), lname + "bottom_up_blocks." + std::to_string(index) + ".", outch * 2, outch, 1, 5, 1);  
    return inner_out;
}


ILayer* top_down(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& feat_heigh, ITensor& feat_low, 
                std::string lname, int res_w, int res_h, int outch)
{
    auto upsample_feat = upsample(network, feat_heigh, res_h, res_w, outch);
    ITensor* inputTensors[] = { upsample_feat->getOutput(0), &feat_low};
    auto cat_out = network->addConcatenation(inputTensors, 2);

    auto inner_out = GhostBlocks(network, weightMap, *cat_out->getOutput(0), lname, outch * 2, outch, 1, 5, 1);
 
    return inner_out;
}

ILayer* ConvModule(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, 
        int inch, int outch, int k, int stride, int padding)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{k, k}, weightMap[lname + "conv.weight"], emptywts);
    assert(conv1);   
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setNbGroups(1);
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn", 1e-5);
    auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    return relu;
}

ILayer* invertedRes(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int s) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int branch_features = outch / 2;
    ITensor *x1, *x2i, *x2o;
    if (s > 1) {
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch, DimsHW{3, 3}, weightMap[lname + "branch1.0.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{s, s});
        conv1->setPaddingNd(DimsHW{1, 1});
        conv1->setNbGroups(inch);
        IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "branch1.1", 1e-5);
        IConvolutionLayer* conv2 = network->addConvolutionNd(*bn1->getOutput(0), branch_features, DimsHW{1, 1}, weightMap[lname + "branch1.2.weight"], emptywts);
        assert(conv2);
        IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "branch1.3", 1e-5);
        IActivationLayer* relu1 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
        assert(relu1);
        x1 = relu1->getOutput(0);
        x2i = &input;
    } else {
        Dims d = input.getDimensions();
        ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        ISliceLayer *s2 = network->addSlice(input, Dims3{ d.d[0] / 2, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        x1 = s1->getOutput(0);
        x2i = s2->getOutput(0);
    }

    IConvolutionLayer* conv3 = network->addConvolutionNd(*x2i, branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.0.weight"], emptywts);
    assert(conv3);
    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "branch2.1", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn3->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu2);
    IConvolutionLayer* conv4 = network->addConvolutionNd(*relu2->getOutput(0), branch_features, DimsHW{3, 3}, weightMap[lname + "branch2.3.weight"], emptywts);
    assert(conv4);
    conv4->setStrideNd(DimsHW{s, s});
    conv4->setPaddingNd(DimsHW{1, 1});
    conv4->setNbGroups(branch_features);
    IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "branch2.4", 1e-5);
    IConvolutionLayer* conv5 = network->addConvolutionNd(*bn4->getOutput(0), branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.5.weight"], emptywts);
    assert(conv5);
    IScaleLayer *bn5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), lname + "branch2.6", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn5->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu3);

    ITensor* inputTensors1[] = {x1, relu3->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors1, 2);
    assert(cat1);

    Dims dims = cat1->getOutput(0)->getDimensions();
    IShuffleLayer *sf1 = network->addShuffle(*cat1->getOutput(0));
    assert(sf1);
    sf1->setReshapeDimensions(Dims4(2, dims.d[0] / 2, dims.d[1], dims.d[2]));
    sf1->setSecondTranspose(Permutation{1, 0, 2, 3});

    Dims dims1 = sf1->getOutput(0)->getDimensions();
    IShuffleLayer *sf2 = network->addShuffle(*sf1->getOutput(0));
    assert(sf2);
    sf2->setReshapeDimensions(Dims3(dims.d[0], dims.d[1], dims.d[2]));
    return sf2;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../nanodet-plus-m_416.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};


    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 24, DimsHW{3, 3}, weightMap["backbone.conv1.0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.conv1.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu1);
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});


    ILayer* inputs1 = invertedRes(network, weightMap, *pool1->getOutput(0), "backbone.stage2.0.", 24, 116, 2);
    inputs1 = invertedRes(network, weightMap, *inputs1->getOutput(0), "backbone.stage2.2.", 116, 116, 1);
    inputs1 = invertedRes(network, weightMap, *inputs1->getOutput(0), "backbone.stage2.3.", 116, 116, 1);
    ILayer* inputs2 = invertedRes(network, weightMap, *inputs1->getOutput(0), "backbone.stage3.0.", 116, 232, 2);
    inputs2 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage3.1.", 232, 232, 1);
    inputs2 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage3.2.", 232, 232, 1);
    inputs2 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage3.3.", 232, 232, 1);
    inputs2 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage3.4.", 232, 232, 1);
    inputs2 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage3.5.", 232, 232, 1);
    inputs2 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage3.6.", 232, 232, 1);
    inputs2 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage3.7.", 232, 232, 1);
    ILayer* inputs3 = invertedRes(network, weightMap, *inputs2->getOutput(0), "backbone.stage4.0.", 232, 464, 2);
    inputs3 = invertedRes(network, weightMap, *inputs3->getOutput(0), "backbone.stage4.1.", 464, 464, 1);
    inputs3 = invertedRes(network, weightMap, *inputs3->getOutput(0), "backbone.stage4.2.", 464, 464, 1);
    inputs3 = invertedRes(network, weightMap, *inputs3->getOutput(0), "backbone.stage4.3.", 464, 464, 1);

    ILayer* reduce1 = ConvModule(network, weightMap,*inputs1->getOutput(0), "fpn.reduce_layers.0.", 116, 96,1, 1, 0);   
    ILayer* reduce2 = ConvModule(network, weightMap,*inputs2->getOutput(0), "fpn.reduce_layers.1.", 232, 96,1, 1, 0);
    ILayer* reduce3 = ConvModule(network, weightMap,*inputs3->getOutput(0), "fpn.reduce_layers.2.", 464, 96,1, 1, 0);

    ILayer* inner_out3 = reduce3;
    ILayer* inner_out2 = top_down(network, weightMap,*inner_out3->getOutput(0), *reduce2->getOutput(0), "fpn.top_down_blocks.0.", 26,26,96);
    ILayer* inner_out1 = top_down(network, weightMap,*inner_out2->getOutput(0), *reduce1->getOutput(0), "fpn.top_down_blocks.0.", 52,52,96);

    ILayer* outs1 = inner_out1;
    ILayer* outs2 = bottom_up(network, weightMap,*inner_out2->getOutput(0), *outs1->getOutput(0), "fpn.", 96, 0);
    ILayer* outs3 = bottom_up(network, weightMap,*inner_out3->getOutput(0), *outs2->getOutput(0), "fpn.", 96, 1);
    ILayer* outs4 = extra_lvl(network, weightMap,*reduce3->getOutput(0), *outs3->getOutput(0), "fpn.", 96);

    ILayer* fea1 = plus_head(network, weightMap,*outs1->getOutput(0), "head.", 96, 0);
    ILayer* fea2 = plus_head(network, weightMap,*outs2->getOutput(0), "head.", 96, 1);
    ILayer* fea3 = plus_head(network, weightMap,*outs3->getOutput(0), "head.", 96, 2);
    ILayer* fea4 = plus_head(network, weightMap,*outs4->getOutput(0), "head.", 96, 3);

    ITensor* inputTensors[] = {fea1->getOutput(0), fea2->getOutput(0), fea3->getOutput(0), fea4->getOutput(0)};
    auto results = network->addConcatenation(inputTensors, 4);    

    results->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*results->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./nanodet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./nanodet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("nanodet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("nanodet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
