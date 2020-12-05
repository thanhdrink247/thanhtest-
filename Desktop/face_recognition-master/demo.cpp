#include "src/face_recognition.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <faiss/IndexFlat.h>
using idx_t = faiss::Index::idx_t;

void video_cap() {
    std::string ImagesPath = "/home/code/Pictures/hd/face";

    std::string detectorParamPath = "../models/retina.param";
    std::string detectorBinPath = "../models/retina.bin";
    std::string recognizerParamPath = "../models/f34.param";
    std::string recognizerBinPath = "../models/f34.bin";
    FaceRecognizer faceDetector = FaceRecognizer(detectorParamPath, detectorBinPath,
                                                 recognizerParamPath, recognizerBinPath);
    int d = 128;                            // dimension
    int nb = 200000;                       // database size
    FILE  *fptr;
    if ((fptr = fopen("/home/code/Pictures/hd/face/face.txt","r")) == NULL){
        printf("Error! opening file");

        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    float *xb = new float[d * nb];

    for(int i = 0; i < nb; i++) {
        //printf("\n%d\n",i);
        for(int j = 0; j < d; j++)
        {
            fscanf(fptr,"%f",&xb[d*i+j]);
            //    printf("%8f   ",xb[d*i+j]);

        }

    }
    faiss::IndexFlatL2 index(128);           // call constructor

    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(200000, xb);  // add vectors to the index

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
        faceDetector.Process(frame,index);

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
    int d = 128;                            // dimension
    int nb = 200000;                       // database size
    FILE  *fptr;
    if ((fptr = fopen("/home/code/Pictures/hd/face/face.txt","r")) == NULL){
        printf("Error! opening file");

        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    float *xb = new float[d * nb];

    for(int i = 0; i < nb; i++) {
        //printf("\n%d\n",i);
        for(int j = 0; j < d; j++)
        {
            fscanf(fptr,"%f",&xb[d*i+j]);
        //    printf("%8f   ",xb[d*i+j]);

        }

    }

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
        faceDetector.Process(originImage,xb);
        printf("-------------------------------\n");
*/
     video_cap();
}

