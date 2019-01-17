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

        createTrackbar("Num Corners:", output_title, &num_corners, max_corners, ShiTomasi_Demo);//�]�w�̦h�i�˴��X�h�֨��I
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
    ���I�˴�
    void cv::goodFeaturesToTrack(
            cv::InputArray image, // ��J�Ϲ��]CV_8UC1 CV_32FC1�^
            cv::OutputArray corners, // ��X���Ivector
            int maxCorners, // �̤j���I�ƥ�
            double qualityLevel, // �~����ǫY�ơ]�p��1.0�����ơA�@��b0.01-0.1�����^
            double minDistance, // �̤p�Z���A�p�󦹶Z�����I����
            cv::InputArray mask = noArray(), // mask=0���I����
            int blockSize = 3, // �ϥΪ��F���
            bool useHarrisDetector = false, // false ='Shi Tomasi metric'
            double k = 0.04 // Harris���I�˴��ɨϥ�
        );
        �Ĥ@�ӰѼƬO��J�Ϲ��]8�줸��32���q�D�ϡ^�C

        �ĤG�ӰѼƬO�˴��쪺�Ҧ����I�A������vector�ΰ}�C�A�ѹ�ڵ��w���Ѽ������өw�C�p�G�Ovector�A�������ӬO�@�ӥ]�tcv::Point2f��vector��H�F�p�G�����Ocv::Mat,���򥦪��C�@������@�Ө��I�A�I��x�By��m���O�O��C�C

        �ĤT�ӰѼƥΩ󭭩w�˴��쪺�I�ƪ��̤j�ȡC

        �ĥ|�ӰѼƪ���˴��쪺���I���~����ǡ]�q�`�O0.10��0.01�������ƭȡA����j��1.0�^�C

        �Ĥ��ӰѼƥΩ�Ϥ��۾F��Ө��I���̤p�Z���]�p��o�ӶZ���o�I�N�i��X�֡^�C

        �Ĥ��ӰѼƬOmask�A�p�G���w�A�������ץ����M��J�Ϲ��@�P�A�B�bmask�Ȭ�0�B���i�樤�I�˴��C

        �ĤC�ӰѼƬOblockSize�A��ܦb�p�⨤�I�ɰѻP�B�⪺�ϰ�j�p�A�`�έȬ�3�A���O�p�G�Ϲ����ѪR�׸����h�i�H�Ҽ{�ϥθ��j�@�I���ȡC

        �ĤK�ӰѼƥΩ���w���I�˴�����k�A�p�G�Otrue�h�ϥ�Harris���I�˴��Afalse�h�ϥ�Shi Tomasi�t��k�C

        �ĤE�ӰѼƬO�b�ϥ�Harris�t��k�ɨϥΡA�̦n�ϥιw�]��0.04�C
	*/
	goodFeaturesToTrack(gray_src, corners, num_corners, qualityLevel, minDistance, Mat(), blockSize, useHarris, k);

	printf("Number of Detected Corners:  %d\n", corners.size());

	for (size_t t = 0; t < corners.size(); t++) {
		circle(resultImg, corners[t], 2, Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow(output_title, resultImg);
}
