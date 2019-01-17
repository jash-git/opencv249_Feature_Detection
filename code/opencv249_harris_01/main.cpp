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
int thresh = 70;
int max_count = 255;
const char* output_title = "HarrisCornerDetection Result";
void Harris_Demo(int, void*);
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

        createTrackbar("Threshold:", output_title, &thresh, max_count, Harris_Demo);
        Harris_Demo(0, 0);
    }
    waitKey(0);
    Pause();
    return 0;
}
void Harris_Demo(int, void*) {
	Mat dst, norm_dst, normScaleDst;

	dst = Mat::zeros(gray_src.size(), CV_32FC1);

	int blockSize = 2;
	int ksize = 3;
	double k = 0.04;
	/*
    Harris���I�˴�
    void cornerHarris(InputArray src, OutputArray dst, int blockSize, int ksize, double k, int borderType=BORDER_DEFAULT)

        src�G��J�ϡA8�줸�ίB�I�Ƴ�q�D�ϡC
        dst�G��X�ϡA�x�sHarris�˴����G�A���A��CV_32FC1�A�ؤo�M��J�ϬۦP�C
        blockSize�G�۾F�������ؤo�C
        ksize�GSobel��l���o�i���ҪO�j�p�C
        k�GHarris�ѼơA�Y���U����{����k�ȡC
        borderType�G��t�X�R�覡�C
	*/
	cornerHarris(gray_src, dst, blockSize, ksize, k, BORDER_DEFAULT);
	/*
    �k�@�ƨ禡
    normalize(src, dst, alpha, beta, norm_type, dtype, mask)

        src-��J�}�C�C

        dst-�PSRC�j�p�ۦP����X�}�C�C

        �\-�d�ƭȦb�d���k�@�ƪ����p�U�k�@�ƨ���C���d����ɡC

        �]-�W���d��b�d���k�@�ƪ����p�U�F�����Ω�d���k�@�ơC

        ���W��-�W�d�ƫ��O�]���U�����Ӹ`�^�C
            NORM_MINMAX: �}�C���ƭȳQ�������Y���@�ӫ��w���d��A�u���k�@�ơC
            NORM_INF: �k�@�ư}�C���]���񳷤ҶZ���^L�۽d��(����Ȫ��̤j��)
            NORM_L1:  �k�@�ư}�C���]�ҫ��y�Z���^L1-�d��(����Ȫ��M)
            NORM_L2: �k�@�ư}�C��(�ڴX���w�Z��)L2-�d��

        dType�X�X���X���t�ɡA��X�}�C�㦳�PSRC�ۦP�����O�F�_�h�A���㦳�PSRC�ۦP���q�D�ƩM�`�ס�CVH-MatthAsHead�]DyType�^�C
	*/
	normalize(dst, norm_dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	/*
    �ഫ�줸
    �p���J�ϦU�����A�ñN���G�ন8�줸��
    void convertScaleAbs(InputArray src, OutputArray dst, double alpha=1, double beta=0)

        src�G��J�ϡC
        dst�G��X�ϡC
        alpha�G��ܩʪ����k�]�l�C
        beta�G��ܩʪ��[�k�]�l�C
        ���禡�D�n�i��3�B�J�F1.�p�� 2.������� 3.�ন�L���t��8�줸��
	*/
	convertScaleAbs(norm_dst, normScaleDst);

	Mat resultImg = src.clone();

	for (int row = 0; row < resultImg.rows; row++) {
		uchar* currentRow = normScaleDst.ptr(row);//���Ф覡����Ƕ�����
		for (int col = 0; col < resultImg.cols; col++) {
			int value = (int)*currentRow;//���Ф覡����Ƕ�����
			if (value > thresh) {
				circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);
			}
			currentRow++;
		}
	}

	imshow(output_title, resultImg);
}
