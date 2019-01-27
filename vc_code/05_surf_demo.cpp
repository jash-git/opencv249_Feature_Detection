#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char** argv) {
	Mat src = imread("D:/vcprojects/images/test.png", IMREAD_GRAYSCALE);
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);

	// SURF特征检测
	int minHessian = 100;
	Ptr<SURF> detector = SURF::create(minHessian);
	vector<KeyPoint> keypoints;
	detector->detect(src, keypoints, Mat());

	// 绘制关键点
	Mat keypoint_img;
	drawKeypoints(src, keypoints, keypoint_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("KeyPoints Image", keypoint_img);

	waitKey(0);
	return 0;
}