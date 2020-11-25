//
// Created by d on 22/10/2020.
//

#ifndef FACE_DETECTION_FACE_RECOGNITION_H
#define FACE_DETECTION_FACE_RECOGNITION_H

#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include "aligner.h"
#include <string>
#include <experimental/filesystem>
#include "opencv2/highgui.hpp"

class FaceRecognizer {
public:
    int modelSize = 640;
    float pixel_mean[3] = {0, 0, 0};
    float pixel_std[3] = {1, 1, 1};
    float scale;
    std::vector<cv::Point2f> landmarks;
    ncnn::Net R50RetinaFace;
    ncnn::Net R100ArcFace;

    FaceRecognizer(std::string detectorParamPath, std::string detectorBinPath,
                   std::string recognizerParamPath, std::string recognizerBinPath) {
        R50RetinaFace.load_param(detectorParamPath.data());
        R50RetinaFace.load_model(detectorBinPath.data());
        R100ArcFace.load_param(recognizerParamPath.data());
        R100ArcFace.load_model(recognizerBinPath.data());
    }

    cv::Mat PreProcess(const cv::Mat &img) {
        cv::Mat scaledImage;
        float long_side = std::max(img.cols, img.rows);
        scale = modelSize / long_side;
        cv::resize(img, scaledImage, cv::Size(img.cols * scale, img.rows * scale));
        return scaledImage;
    }


    bool Detect(const cv::Mat &scaledImage, cv::Mat originImage) {
        landmarks.clear();
        ncnn::Mat input = ncnn::Mat::from_pixels_resize(
                scaledImage.data,
                ncnn::Mat::PIXEL_BGR2RGB,
                scaledImage.cols, scaledImage.rows,
                scaledImage.cols, scaledImage.rows
        );
        input.substract_mean_normalize(pixel_mean, pixel_std);
        ncnn::Extractor _extractor = R50RetinaFace.create_extractor();
        _extractor.input("data", input);

        std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            int stride = _feat_stride_fpn[i];
            ac[i].Init(stride, anchor_cfg[stride], false);
        }

        std::vector<Anchor> proposals;
        proposals.clear();

        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            ncnn::Mat cls;
            ncnn::Mat reg;
            ncnn::Mat pts;
            char clsname[100];
            sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
            char regname[100];
            sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
            char ptsname[100];
            sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
            _extractor.extract(clsname, cls);
            _extractor.extract(regname, reg);
            _extractor.extract(ptsname, pts);
            ac[i].FilterAnchor(cls, reg, pts, proposals);
        }
        std::vector<Anchor> result;
        nms_cpu(proposals, nms_threshold, result);
        if (!result.empty()) {
            for (int j = 0; j < result[0].pts.size(); ++j) {
                cv::Point point = cv::Point(
                        result[0].pts[j].x / scale,
                        result[0].pts[j].y / scale
                );
                landmarks.emplace_back(point);
            }
        }
        for (int i = 0; i < result.size(); i++) {
            cv::rectangle(
                    scaledImage,
                    cv::Point(result[i].finalbox.x, result[i].finalbox.y),
                    cv::Point(result[i].finalbox.width, result[i].finalbox.height),
                    (0, 255, 255),
                    2
            );

        }
        https://nguyenvanhieu.vn/ham-trong-c/
        cv::imshow("img", scaledImage);
        cv::waitKey(1);

        return !result.empty();
    }

    std::vector<uint8_t> Recognize(cv::Mat alignedFace) const {
        ncnn::Mat input = ncnn::Mat::from_pixels(
                alignedFace.data,
                ncnn::Mat::PIXEL_BGR2RGB,
                alignedFace.cols, alignedFace.rows
        );
        ncnn::Extractor ex = R100ArcFace.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", input);
        ncnn::Mat out;
        ex.extract("fc1", out);
        uint8_t *pixels = reinterpret_cast<uint8_t *>(out.data);
        int data_size = out.w * sizeof(float);
        std::vector<uint8_t> result(data_size);
        memcpy(result.data(), pixels, data_size);
        return result;
    }

    std::pair<std::vector<uint8_t>, std::vector<float>> Process(cv::Mat originImage) {
        std::vector<uint8_t> featureVector;
        std::vector<float> eyes;
        cv::Mat scaledImg = this->PreProcess(originImage);
        bool isDetected = this->Detect(scaledImg, originImage);
        if (!isDetected) {
            cv::rotate(originImage, originImage, cv::ROTATE_180);
            cv::Mat scaledImg = this->PreProcess(originImage);
            isDetected = this->Detect(scaledImg, originImage);
        }
        if (isDetected) {
            cv::Mat alignedFace;
            Aligner aligner;
            aligner.AlignFace(originImage, this->landmarks, &alignedFace);
            eyes.push_back(this->landmarks[0].x);
            eyes.push_back(this->landmarks[0].y);
            eyes.push_back(this->landmarks[1].x);
            eyes.push_back(this->landmarks[1].y);
            featureVector = this->Recognize(alignedFace);

            //cv::imshow("ori",  originImage);
            //  cv::waitKey(500);
        }
        return std::make_pair(featureVector, eyes);
    }

};

#endif //FACE_DETECTION_FACE_RECOGNITION_H
