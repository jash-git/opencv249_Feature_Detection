#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
//3.1的寫法- #include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/objdetect/objdetect.hpp"//HOG

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
	src = imread("input.jpg");
	if (!src.data)
    {
		printf("could not load image...\n");
	}
    else
    {
        namedWindow("input image", CV_WINDOW_AUTOSIZE);
        imshow("input image", src);
        cvtColor(src, gray_src, COLOR_BGR2GRAY);

        /*Mat dst, dst_gray;
        resize(src, dst, Size(64, 128));
        cvtColor(dst, dst_gray, COLOR_BGR2GRAY);
        HOGDescriptor detector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

        vector<float> descriptors;
        vector<Point> locations;
        detector.compute(dst_gray, descriptors, Size(0, 0), Size(0, 0), locations);
        printf("number of HOG descriptors : %d", descriptors.size());
        */
        HOGDescriptor hog = HOGDescriptor();
        hog.setSVMDetector(hog.getDefaultPeopleDetector());

        vector<Rect> foundLocations;
        hog.detectMultiScale(src, foundLocations, 0, Size(8, 8), Size(32, 32), 1.05, 2);
        Mat result = src.clone();

        for (size_t t = 0; t < foundLocations.size(); t++)
        {
            rectangle(result, foundLocations[t], Scalar(0, 0, 255), 2, 8, 0);
        }

        namedWindow("HOG SVM Detector Demo", CV_WINDOW_AUTOSIZE);
        imshow("HOG SVM Detector Demo", result);
    }
    waitKey(0);
    Pause();
    return 0;
}

