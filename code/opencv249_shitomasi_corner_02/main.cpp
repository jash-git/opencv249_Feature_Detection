#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <cstdio>

#include <sys/timeb.h>
#if defined(WIN32)
    #define  TIMEB    _timeb
    #define  ftime    _ftime
    typedef __int64 TIME_T;
#else
    #define TIMEB timeb
    typedef long long TIME_T;
#endif

using namespace cv;
using namespace std;

void Pause()
{
    printf("Press Enter key to continue...");
    fgetc(stdin);
}

Mat src, gray_src;
int num_corners = 25;
int max_corners = 200;
const char* output_title = "ShiTomasi Detector";
void ShiTomasi_Demo(int, void*);
int main()
{
	src = imread("input.png");
	if (!src.data)
    {
		printf("could not load image...\n");
	}
    else
    {
        namedWindow("input image", CV_WINDOW_AUTOSIZE);
        imshow("input image", src);

        namedWindow(output_title, CV_WINDOW_AUTOSIZE);
        cvtColor(src, gray_src, COLOR_BGR2GRAY);
        imshow("gray_src", gray_src);

        createTrackbar("Num Corners:", output_title, &num_corners, max_corners, ShiTomasi_Demo);//設定最多可檢測出多少角點
        ShiTomasi_Demo(0, 0);
    }
    waitKey(0);
    Pause();
    return 0;
}

void ShiTomasi_Demo(int, void*) {
	if (num_corners < 5) {
		num_corners = 5;
	}

	vector <Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarris = false;
	double k = 0.04;
	Mat resultImg ;
	src.copyTo(resultImg); //Mat resultImg = src.clone();
	//cvtColor(resultImg, resultImg, COLOR_GRAY2BGR);
	/*
    角點檢測
    void cv::goodFeaturesToTrack(
            cv::InputArray image, // 輸入圖像（CV_8UC1 CV_32FC1）
            cv::OutputArray corners, // 輸出角點vector
            int maxCorners, // 最大角點數目
            double qualityLevel, // 品質水準係數（小於1.0的正數，一般在0.01-0.1之間）
            double minDistance, // 最小距離，小於此距離的點忽略
            cv::InputArray mask = noArray(), // mask=0的點忽略
            int blockSize = 3, // 使用的鄰域數
            bool useHarrisDetector = false, // false ='Shi Tomasi metric'
            double k = 0.04 // Harris角點檢測時使用
        );
        第一個參數是輸入圖像（8位元或32位單通道圖）。

        第二個參數是檢測到的所有角點，類型為vector或陣列，由實際給定的參數類型而定。如果是vector，那麼它應該是一個包含cv::Point2f的vector對象；如果類型是cv::Mat,那麼它的每一行對應一個角點，點的x、y位置分別是兩列。

        第三個參數用於限定檢測到的點數的最大值。

        第四個參數表示檢測到的角點的品質水準（通常是0.10到0.01之間的數值，不能大於1.0）。

        第五個參數用於區分相鄰兩個角點的最小距離（小於這個距離得點將進行合併）。

        第六個參數是mask，如果指定，它的維度必須和輸入圖像一致，且在mask值為0處不進行角點檢測。

        第七個參數是blockSize，表示在計算角點時參與運算的區域大小，常用值為3，但是如果圖像的解析度較高則可以考慮使用較大一點的值。

        第八個參數用於指定角點檢測的方法，如果是true則使用Harris角點檢測，false則使用Shi Tomasi演算法。

        第九個參數是在使用Harris演算法時使用，最好使用預設值0.04。
	*/
	goodFeaturesToTrack(gray_src, corners, num_corners, qualityLevel, minDistance, Mat(), blockSize, useHarris, k);

	printf("Number of Detected Corners:  %d\n", corners.size());

	for (size_t t = 0; t < corners.size(); t++) {
		circle(resultImg, corners[t], 2, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow(output_title, resultImg);
}
