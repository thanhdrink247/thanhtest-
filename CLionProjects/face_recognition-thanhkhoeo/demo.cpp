#include "src/face_recognition.h"

void video_cap() {
    std::string detectorParamPath = "../models/retina.param";
    std::string detectorBinPath = "../models/retina.bin";
    std::string recognizerParamPath = "../models/f34.param";
    std::string recognizerBinPath = "../models/f34.bin";
    FaceRecognizer faceDetector = FaceRecognizer(detectorParamPath, detectorBinPath,
                                                 recognizerParamPath, recognizerBinPath);
    cv::Mat frame;
    cv::VideoCapture cap("/home/code/Desktop/2.mp4");
    if (!cap.isOpened()) {
        std::cout << "Error can't find the file" << std::endl;
    }  while (cap.read(frame)) {
        cv::Mat alignedFace;
        // cv::Mat originImage = cv::imread(entry.path().string());
        //cv::Mat originImage = cv::imread("/home/code/Pictures/hd/face/bd2.png");
        if (!frame.data) {
            printf("load error");
            throw std::exception();
        }
        faceDetector.Process(frame);

        printf("\n");
        printf("-------------------------------\n");

        //  cv::waitKey(33);
    }
}

 int main() {
/*
    std::string ImagesPath = "/home/code/Pictures/hd/face";

    std::string detectorParamPath = "../models/retina.param";
    std::string detectorBinPath = "../models/retina.bin";
    std::string recognizerParamPath = "../models/f34.param";
    std::string recognizerBinPath = "../models/f34.bin";
    FaceRecognizer faceDetector = FaceRecognizer(detectorParamPath, detectorBinPath,
                                                 recognizerParamPath, recognizerBinPath);


   // for (const auto &entry : std::experimental::filesystem::directory_iterator(ImagesPath)) {
        cv::Mat alignedFace;
       // cv::Mat originImage = cv::imread(entry.path().string());
        //cv::Mat originImage = cv::imread("/home/code/Pictures/hd/face/baby1.jpg");

        // cv::Mat originImage = cv::imread(entry.path().string());
        cv::Mat originImage = cv::imread("/home/code/Pictures/hd/face/bd2.png");
        if (!originImage.data) {
            printf("load error");
            throw std::exception();
        }
        faceDetector.Process(originImage);
        printf("-------------------------------\n");
*/
     video_cap();

}

