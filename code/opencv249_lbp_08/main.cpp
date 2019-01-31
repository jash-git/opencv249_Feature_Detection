#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
//3.1���g�k- #include <opencv2/xfeatures2d.hpp>
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
//3.1���g�k- using namespace cv::xfeatures2d;
using namespace std;

void Pause()
{
    printf("Press Enter key to continue...");
    fgetc(stdin);
}

Mat src, gray_src;
int current_radius = 3;
int max_count = 20;
void ELBP_Demo(int, void*);
int main()
{
	src = imread("input.jpg");
	if (!src.data)
    {
		printf("could not load image...\n");
	}
    else
    {
        const char* output_tt = "LBP Result";
        namedWindow("input image", CV_WINDOW_AUTOSIZE);
        namedWindow(output_tt, CV_WINDOW_AUTOSIZE);
        imshow("input image", src);

        // convert to gray
        cvtColor(src, gray_src, COLOR_BGR2GRAY);
        int width = gray_src.cols;
        int height = gray_src.rows;

        // ����LBP��ʾ
        Mat lbpImage = Mat::zeros(gray_src.rows - 2, gray_src.cols - 2, CV_8UC1);
        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                uchar c = gray_src.at<uchar>(row, col);
                uchar code = 0;
                code |= (gray_src.at<uchar>(row - 1, col - 1) > c) << 7;
                code |= (gray_src.at<uchar>(row - 1, col) > c) << 6;
                code |= (gray_src.at<uchar>(row - 1, col + 1) > c) << 5;
                code |= (gray_src.at<uchar>(row, col + 1) > c) << 4;
                code |= (gray_src.at<uchar>(row + 1, col + 1) > c) << 3;
                code |= (gray_src.at<uchar>(row + 1, col) > c) << 2;
                code |= (gray_src.at<uchar>(row + 1, col - 1) > c) << 1;
                code |= (gray_src.at<uchar>(row, col - 1) > c) << 0;
                lbpImage.at<uchar>(row - 1, col - 1) = code;
            }
        }
        imshow(output_tt, lbpImage);

        // ELBP
        namedWindow("ELBP Result", CV_WINDOW_AUTOSIZE);
        createTrackbar("ELBP Radius:", "ELBP Result", &current_radius, max_count, ELBP_Demo);
        ELBP_Demo(0, 0);
    }
    waitKey(0);
    Pause();
    return 0;
}
void ELBP_Demo(int, void*) {
	int offset = current_radius * 2;
	Mat elbpImage = Mat::zeros(gray_src.rows - offset, gray_src.cols - offset, CV_8UC1);
	int width = gray_src.cols;
	int height = gray_src.rows;

	int numNeighbors = 8;
	for (int n = 0; n < numNeighbors; n++) {
		float x = static_cast<float>(current_radius) * cos(2.0 * CV_PI*n / static_cast<float>(numNeighbors));
		float y = static_cast<float>(current_radius) * -sin(2.0 * CV_PI*n / static_cast<float>(numNeighbors));

		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));

		float ty = y - fy;
		float tx = x - fx;

		float w1 = (1 - tx)*(1 - ty);
		float w2 = tx*(1 - ty);
		float w3 = (1 - tx)* ty;
		float w4 = tx*ty;

		for (int row = current_radius; row < (height - current_radius); row++) {
			for (int col = current_radius; col < (width - current_radius); col++) {
				float t = w1* gray_src.at<uchar>(row + fy, col + fx) + w2* gray_src.at<uchar>(row + fy, col + cx) +
					w3* gray_src.at<uchar>(row + cy, col + fx) + w4* gray_src.at<uchar>(row + cy, col + cx);
				elbpImage.at<uchar>(row- current_radius, col- current_radius) +=
					((t > gray_src.at<uchar>(row, col)) && (abs(t - gray_src.at<uchar>(row, col)) > std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	imshow("ELBP Result", elbpImage);
	return;
}
