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

int max_corners = 20;
int max_count = 500;
Mat src, gray_src;
const char* output_title = "SubPixel Result";
void SubPixel_Demo(int, void*);
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
        namedWindow(output_title, CV_WINDOW_AUTOSIZE);
        createTrackbar("Corners:", output_title, &max_corners, max_count, SubPixel_Demo);
        SubPixel_Demo(0, 0);
    }
    waitKey(0);
    Pause();
    return 0;
}

void SubPixel_Demo(int, void*) {
	if (max_corners < 5) {
		max_corners = 5;
	}
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	double k = 0.04;

	goodFeaturesToTrack(gray_src, corners, max_corners, qualityLevel, minDistance, Mat(), blockSize, false, k);
	cout << "number of corners: " << corners.size() << endl;

	Mat resultImg = src.clone();
	for (size_t t = 0; t < corners.size(); t++) {
		circle(resultImg, corners[t], 2, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow(output_title, resultImg);

	Size winSize = Size(5, 5);
	Size zerozone = Size(-1, -1);
	TermCriteria tc = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);
	cornerSubPix(gray_src, corners, winSize, zerozone, tc);

	for (size_t t = 0; t < corners.size(); t++) {
		cout << (t + 1) << " .point[x, y] = " << corners[t].x << " , " << corners[t].y << endl;
	}
	return;
}
