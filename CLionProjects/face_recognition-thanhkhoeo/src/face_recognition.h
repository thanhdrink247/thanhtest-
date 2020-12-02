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
    std::vector<cv::Point2f>landmarks[100];
    ncnn::Net R50RetinaFace;
    ncnn::Net R100ArcFace;
    int  sizeResult;
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
        for(int i=0; i<100; i++)
        landmarks[i].clear();
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

            for (int i = 0; i < result.size(); i++) {
                for (int j = 0; j < result[i].pts.size(); ++j) {
                    cv::Point point = cv::Point(
                            result[i].pts[j].x / scale,
                            result[i].pts[j].y / scale
                    );
         //           printf(" %f , %f , " , result[i].pts[j].x / scale , result[i].pts[j].x / scale);
                    landmarks[i].emplace_back(point);
                    }
           //     printf("\n");
            }

        }
        sizeResult = result.size();
        for (int i = 0; i < result.size(); i++) {
            cv::rectangle(
                    scaledImage,
                    cv::Point(result[i].finalbox.x, result[i].finalbox.y),
                    cv::Point(result[i].finalbox.width, result[i].finalbox.height),
                    (0, 255, 255),
                    2
            );

        }
        cv::imshow("img", scaledImage);
        cv::waitKey(1);

        return !result.empty();
    }

    std::vector<float> Recognize(cv::Mat alignedFace) const {
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
        std::vector<float> result;

        int dataSize = out.channel(0).w;
        for (int j = 0; j < dataSize; j++) {
            result.push_back(out.channel(0)[j]);
        }
        return result;
    }

    void  Process( cv::Mat originImage ) {
        std::vector<float> featureVector;
        cv::Mat scaledImg = this->PreProcess(originImage);
        bool isDetected = this->Detect(scaledImg, originImage);
        if (isDetected) {
           for (int i = 0; i <sizeResult;++i) {
                cv::Mat alignedFace;
                Aligner aligner;
                aligner.AlignFace(originImage, this->landmarks[i], &alignedFace);
                featureVector = this->Recognize(alignedFace);
                for (auto const &c : featureVector) {
                    printf("%f ,", c);
                }
                printf("\n");

                }//cv::imshow("ori",  originImage);
            //  cv::waitKey(500);
        }
      //  return featureVector;
    }

};

#endif //FACE_DETECTION_FACE_RECOGNITION_H

