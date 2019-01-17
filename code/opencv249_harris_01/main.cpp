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
    Harris角點檢測
    void cornerHarris(InputArray src, OutputArray dst, int blockSize, int ksize, double k, int borderType=BORDER_DEFAULT)

        src：輸入圖，8位元或浮點數單通道圖。
        dst：輸出圖，儲存Harris檢測結果，型態為CV_32FC1，尺寸和輸入圖相同。
        blockSize：相鄰像素的尺寸。
        ksize：Sobel算子的濾波器模板大小。
        k：Harris參數，即為下面方程式的k值。
        borderType：邊緣擴充方式。
	*/
	cornerHarris(gray_src, dst, blockSize, ksize, k, BORDER_DEFAULT);
	/*
    歸一化函式
    normalize(src, dst, alpha, beta, norm_type, dtype, mask)

        src-輸入陣列。

        dst-與SRC大小相同的輸出陣列。

        α-範數值在範圍歸一化的情況下歸一化到較低的範圍邊界。

        β-上限範圍在範圍歸一化的情況下；它不用於範數歸一化。

        正規化-規範化型別（見下面的細節）。
            NORM_MINMAX: 陣列的數值被平移或縮放到一個指定的範圍，線性歸一化。
            NORM_INF: 歸一化陣列的（切比雪夫距離）L∞範數(絕對值的最大值)
            NORM_L1:  歸一化陣列的（曼哈頓距離）L1-範數(絕對值的和)
            NORM_L2: 歸一化陣列的(歐幾里德距離)L2-範數

        dType——當輸出為負時，輸出陣列具有與SRC相同的型別；否則，它具有與SRC相同的通道數和深度＝CVH-MatthAsHead（DyType）。
	*/
	normalize(dst, norm_dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	/*
    轉換位元
    計算輸入圖各像素，並將結果轉成8位元圖
    void convertScaleAbs(InputArray src, OutputArray dst, double alpha=1, double beta=0)

        src：輸入圖。
        dst：輸出圖。
        alpha：選擇性的乘法因子。
        beta：選擇性的加法因子。
        此函式主要進行3步驟；1.計算 2.取絕對值 3.轉成無正負號8位元圖
	*/
	convertScaleAbs(norm_dst, normScaleDst);

	Mat resultImg = src.clone();

	for (int row = 0; row < resultImg.rows; row++) {
		uchar* currentRow = normScaleDst.ptr(row);//指標方式抓取灰階像素
		for (int col = 0; col < resultImg.cols; col++) {
			int value = (int)*currentRow;//指標方式抓取灰階像素
			if (value > thresh) {
				circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);
			}
			currentRow++;
		}
	}

	imshow(output_title, resultImg);
}
