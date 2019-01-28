#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
//3.1的寫法- #include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

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
//3.1的寫法- using namespace cv::xfeatures2d;
using namespace std;

void Pause()
{
    printf("Press Enter key to continue...");
    fgetc(stdin);
}

Mat src, gray_src;

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
        cvtColor(src, gray_src, COLOR_BGR2GRAY);

        int numFeatures = 400;
        initModule_nonfree();//初始化模块，使用SIFT或SURF时用到

        SiftFeatureDetector detector(numFeatures);//3.1的寫法- Ptr<SIFT> detector = SIFT::create(numFeatures);
        vector<KeyPoint> keypoints;
        detector.detect(gray_src, keypoints, Mat());//3.1的寫法- detector->detect(gray_src, keypoints, Mat());

        Mat keypoint_img;
        drawKeypoints(src, keypoints, keypoint_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        namedWindow("SIFT KeyPoints Image", CV_WINDOW_AUTOSIZE);
        imshow("SIFT KeyPoints Image", keypoint_img);
    }
    waitKey(0);
    Pause();
    return 0;
}

