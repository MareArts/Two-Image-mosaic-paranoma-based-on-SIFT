
#include <iostream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>


using namespace std;

typedef struct MATCH_PAIR
{
	int nA;
	int nB;
} MATCH_PAIR;


void MergeImages(IplImage* Image1, IplImage* Image2, IplImage* dstImage);
int FindMatchingPoints(const CvSeq* tKeypoints, const CvSeq* tDescriptors, const CvSeq* srcKeypoints, const CvSeq* srcDescriptors, int descriptor_size, MATCH_PAIR *pMatchPair);
int FindNearestPoints(const float* pA, int laplacian, const CvSeq* srcKeypoints, const CvSeq* srcDescriptors, int descriptor_size);

void main()
{
	
	///////////////////////////////////////////////////////////////////////////////////////////
	//두 영상 읽기
	// 첫 번째 Image read
	IplImage *T1Img, *mT1Img;
	if((T1Img = cvLoadImage("T1.jpg", CV_LOAD_IMAGE_GRAYSCALE)) == NULL)
	{
		printf("A Img open error\n");
		return ;
	}
	//정보 뿌릴 이미지
	mT1Img = cvCreateImage(cvSize(T1Img->width, T1Img->height),T1Img->depth, 3);
	cvCvtColor(T1Img, mT1Img, CV_GRAY2BGR);
	
	// 두 번째 Image read
	IplImage* T2Img, *mT2Img;
	if((T2Img = cvLoadImage("T2.jpg", CV_LOAD_IMAGE_GRAYSCALE)) == NULL)
	{
		printf("B Img open error\n");
		return ;
	}
	//정보 뿌릴 이미지
	mT2Img = cvCreateImage(cvSize(T2Img->width, T2Img->height),T2Img->depth, 3);
	cvCvtColor(T2Img, mT2Img, CV_GRAY2BGR);

	//창만들기
	cvNamedWindow("Origin T1 Img", 1);
	cvNamedWindow("Origin T2 Img", 1);
	
	cvShowImage("Origin T1 Img", T1Img);
	cvShowImage("Origin T2 Img", T2Img);
	///////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////////////////////////////////
	//surf 특징점 찾기
	CvMemStorage* storage =cvCreateMemStorage(0);
	CvSURFParams params = cvSURFParams(300, 0);

	//T1 Img
	CvSeq *T1_Keypoints = NULL;
	CvSeq *T1_Descriptors = NULL;
	cvExtractSURF(T1Img, NULL, &T1_Keypoints, &T1_Descriptors, storage, params);
	printf("T1 Img - Keypoint: %d\n", T1_Keypoints->total);
	printf("T1 Img - Descriptors : %d\n", T1_Descriptors->total);

	//T2 Img
	CvSeq *T2_Keypoints = NULL;
	CvSeq *T2_Descriptors = NULL;
	cvExtractSURF(T2Img, NULL, &T2_Keypoints, &T2_Descriptors, storage, params);
	printf("T2 Img - Keypoints : %d\n", T2_Keypoints->total);
	printf("T2 Img - Descriptors : %d\n", T2_Descriptors->total);
	
	//특징점 뿌리기 1
	CvSURFPoint* surf1;
	for(int i = 0; i < (T1_Keypoints ? T1_Keypoints->total : 0); i++ )
    {
		surf1 = (CvSURFPoint*) cvGetSeqElem(T1_Keypoints, i);
		int r = surf1->size/2;
		cvCircle( mT1Img, cvPoint(surf1->pt.x, surf1->pt.y) , r, CV_RGB(0,255,0));
		cvLine( mT1Img, cvPoint(surf1->pt.x + r, surf1->pt.y + r), cvPoint(surf1->pt.x - r, surf1->pt.y - r), CV_RGB(0,255,0));
        cvLine( mT1Img, cvPoint(surf1->pt.x - r, surf1->pt.y + r), cvPoint(surf1->pt.x + r, surf1->pt.y - r), CV_RGB(0,255,0));
    }

	//특징점 뿌리기 2
	CvSURFPoint* surf2;
	for(int i = 0; i < (T2_Keypoints ? T2_Keypoints->total : 0); i++ )
    {
		surf2 = (CvSURFPoint*) cvGetSeqElem(T2_Keypoints, i);
		int r = surf2->size/2;
		cvCircle( mT2Img, cvPoint(surf2->pt.x, surf2->pt.y) , r, CV_RGB(0,255,0));
		cvLine( mT2Img, cvPoint(surf2->pt.x + r, surf2->pt.y + r), cvPoint(surf2->pt.x - r, surf2->pt.y - r), CV_RGB(0,255,0));
        cvLine( mT2Img, cvPoint(surf2->pt.x - r, surf2->pt.y + r), cvPoint(surf2->pt.x + r, surf2->pt.y - r), CV_RGB(0,255,0));
    }

	//surf 특징점 뿌리기
	cvNamedWindow("Surf T1 Img", 1);
	cvNamedWindow("Surf T2 Img", 1);
	
	cvShowImage("Surf T1 Img", mT1Img);
	cvShowImage("Surf T2 Img", mT2Img);
	///////////////////////////////////////////////////////////////////////////////////////////
	

	///////////////////////////////////////////////////////////////////////////////////////////
	//Image 병합
	IplImage* MergeImg = cvCreateImage(cvSize(T1Img->width+T2Img->width, T1Img->height), IPL_DEPTH_8U, 3);
	MergeImages(T1Img, T2Img, MergeImg);	
	///////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////////////////////////////////
	//매칭
	MATCH_PAIR *pMatchPair = new MATCH_PAIR[T1_Keypoints->total];
	int descriptor_size = params.extended? 128 : 64;
	int nMatchingCount = FindMatchingPoints(T1_Keypoints, T1_Descriptors, T2_Keypoints, T2_Descriptors, descriptor_size, pMatchPair);
	printf("matching count = %d\n", nMatchingCount);

	CvPoint2D32f *pt1 = new CvPoint2D32f[nMatchingCount];
	CvPoint2D32f *pt2 = new CvPoint2D32f[nMatchingCount];

	int x1, y1, x2, y2;
	for(int k=0; k<nMatchingCount; k++)
	{
		//매칭 k번째의 T1, T2에서의 키 포인트 정보 뽑기
		surf1 = (CvSURFPoint*) cvGetSeqElem(T1_Keypoints, pMatchPair[k].nA);
		x1 = cvRound(surf1->pt.x);
		y1 = cvRound(surf1->pt.y);

		surf2 = (CvSURFPoint*) cvGetSeqElem(T2_Keypoints, pMatchPair[k].nB);
		x2 = cvRound(surf2->pt.x) + T1Img->width;
		y2 = cvRound(surf2->pt.y);

		//병합 영상에 라인으로 표시하기
		CvPoint r1 = cvPoint(x1, y1);
		CvPoint r2 = cvPoint(x2, y2);
		cvLine(MergeImg, r1, r2, CV_RGB(0, 0, 255));

		pt1[k] = surf1->pt;
		pt2[k] = surf2->pt;
	}

	//병합 영상 뿌리기
	cvNamedWindow("Merging Img",1);
	cvShowImage("Merging Img", MergeImg);
	///////////////////////////////////////////////////////////////////////////////////////////


	
	///////////////////////////////////////////////////////////////////////////////////////////
	//호모그래피 계산
	CvPoint corners[4]={{0, 0}, {T1Img->width, 0}, {T1Img->width, T1Img->height}, {0, T1Img->height}};
	if(nMatchingCount<4)
	{
		printf("We need more than 4 matching points");
		printf("to calculate a homography transform\n");
		return ;
	}

	CvMat M1, M2;	
	double H[9];
	CvMat mxH = cvMat(3, 3, CV_64F, H);
	M1 = cvMat(1, nMatchingCount, CV_32FC2, pt1);
	M2 = cvMat(1, nMatchingCount, CV_32FC2, pt2);
	if( !cvFindHomography(&M1, &M2, &mxH, CV_RANSAC, 2))
	{
		printf("Find Homography Fail!\n");
		return ;
	}
	
	//호모그래피 출력
	printf(" Homography matrix\n");
	for( int rows=0; rows<3; rows++ )
	{
		for( int cols=0; cols<3; cols++ )
		{
			printf("%lf ", cvmGet(&mxH, rows, cols) );
		}
		printf("\n");
	}
	///////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////////////////////////////////
	//모자이크 영상 만들기
	IplImage* WarpImg = cvCreateImage(cvSize(T1Img->width*2, T1Img->height*2), T1Img->depth, T1Img->nChannels);	
	cvWarpPerspective(T1Img, WarpImg, &mxH);
	cvSetImageROI(WarpImg, cvRect(0, 0, T2Img->width, T2Img->height));
	cvCopy(T2Img, WarpImg);
	cvResetImageROI(WarpImg);
	
	//모자이크 영상 뿌리기
	cvNamedWindow("WarpImg Img",1);
	cvShowImage("WarpImg Img", WarpImg);
	///////////////////////////////////////////////////////////////////////////////////////////	
	cvWaitKey(0);

	delete pMatchPair;
	delete pt1;
	delete pt2;


	cvReleaseImage(&WarpImg);

	cvReleaseImage(&T1Img);
	cvReleaseImage(&T2Img);

	cvReleaseImage(&mT1Img);
	cvReleaseImage(&mT2Img);

	cvReleaseImage(&MergeImg);

}


void MergeImages(IplImage* Image1, IplImage* Image2, IplImage* dstImage)
{
	cvSet(dstImage, CV_RGB(255, 255, 255));
	cvSetImageROI(dstImage, cvRect(0, 0, Image1->width, Image1->height));
	cvSetImageCOI(dstImage, 1); //채널 1
	cvCopy(Image1, dstImage);	
	cvSetImageCOI(dstImage, 2); //채널 2
	cvCopy(Image1, dstImage);	
	cvSetImageCOI(dstImage, 3); //채널 3
	cvCopy(Image1, dstImage);

	
	cvSetImageROI(dstImage, cvRect(Image1->width, 0, Image2->width, Image2->height));
	cvSetImageCOI(dstImage, 1); //채널 1
	cvCopy(Image2, dstImage);
	cvSetImageCOI(dstImage, 2); //채널 2
	cvCopy(Image2, dstImage);
	cvSetImageCOI(dstImage, 3); //채널 3
	cvCopy(Image2, dstImage);
	cvResetImageROI(dstImage);
	
}


int FindMatchingPoints(const CvSeq* tKeypoints, const CvSeq* tDescriptors, const CvSeq* srcKeypoints, const CvSeq* srcDescriptors, int descriptor_size, MATCH_PAIR *pMatchPair)
{
	int i;
	float* pA;
	int nMatchB;
	CvSURFPoint* surfA;
	int k=0;
	for(i=0; i<tDescriptors->total; i++)
	{
		pA = (float*) cvGetSeqElem(tDescriptors, i);
		surfA = (CvSURFPoint*) cvGetSeqElem(tKeypoints, i);
		nMatchB = FindNearestPoints(pA, surfA->laplacian, srcKeypoints, srcDescriptors, descriptor_size);
		if(nMatchB > 0)
		{
			pMatchPair[k].nA=i;
			pMatchPair[k].nB = nMatchB;
			k++;
		}
	}

	return k;
}


int FindNearestPoints(const float* pA, int laplacian, const CvSeq* srcKeypoints, const CvSeq* srcDescriptors, int descriptor_size)
{
	int i, k;
	float* pB;
	CvSURFPoint *surfB;
	int nMatch = -1;
	double sum2, min1 = 10000, min2 = 10000;
	for(i=0; i<srcDescriptors->total; i++)
	{
		surfB = (CvSURFPoint*) cvGetSeqElem(srcKeypoints, i);
		pB = (float*) cvGetSeqElem(srcDescriptors, i);
		if(laplacian != surfB->laplacian)
			continue;

		sum2 = 0.0f;
		for(k=0; k<descriptor_size; k++)	{	sum2 +=(pA[k]-pB[k])*(pA[k]-pB[k]);	}

		if(sum2 < min1)
		{
			min2 = min1;
			min1 = sum2;
			nMatch = i;
		}
		else if(sum2<min2)	{	min2 = sum2;	}
	}
	if(min1<0.6*min2)
		return nMatch;

	return -1;
}


