#include <string>   
#include <cv.h>
#include <highgui\highgui.hpp>
#include <opencv2\opencv.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <features2d\features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <queue>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <limits>
using namespace cv;
using namespace std;

#define N_FRAMES 50
//#define GPU
# define PI           3.14159265358979323846  /* pi */
# define CUT_COLUMN 120


static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color,double line_thickness=1)
{
	for(int y = 0; y < cflowmap.rows; y += step)
		for(int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),color,line_thickness);
			//circle(cflowmap, Point(x,y), 2, color, -1);
		}
}

static void drawOptFlowColumn(const Mat& flow, Mat& cflowmap, int column,int step, const Scalar& color,double line_thickness=1)
{
	for(int y = 0; y < cflowmap.rows; y += step)
	{
		const Point2f& fxy = flow.at<Point2f>(y, column);
		line(cflowmap, Point(column,y), Point(cvRound(column+fxy.x), cvRound(y+fxy.y)),color,line_thickness);
		//circle(cflowmap, Point(x,y), 2, color, -1);
	}
}

void x_component(Mat& input)
{
	for (int row=0;row<input.rows;row++)
	{
		for (int col=0;col<input.cols;col++ )
		{
			//input.at<Point2f>(row, col);
			Point2f& fxy=input.at<Point2f>(row,col);
			fxy.y=0;
			input.at<Point2f>(row,col)=fxy;
		}
	}
}
void save_x_component(Mat& input, Mat& output, int frameCounter)
{
	for (int row=0;row<input.rows;row++)
	{
		for (int col=0;col<input.cols;col++ )
		{
			Point2f& fxy=input.at<Point2f>(row,col);
			fxy.y=0;
			fxy.x=abs(fxy.x);
			output.at<char>(row,frameCounter)=2*fxy.x;
		}
	}
}

int modeEachRow(const Mat& input,float& mode)
{
	//Mat bins=Mat::zeros(input.rows,input.cols,CV_8U);
	int* bins=new int[input.cols];
	memset(bins,0,sizeof(int)*input.cols);
	int no_mode=0,mode_cnt=0,cnt=0,index=0;
	float num;
	for(int col=0;col<input.cols;col++)
	{
		cnt=0;
		num=input.at<float>(0,col);
		for(int i=0;i<input.cols;i++)
		{
			if(num==input.at<float>(0,i))
				cnt++;
		}
		bins[col]=cnt;
		if(mode_cnt<cnt)
		{
			mode_cnt=cnt;
			index=col;
		}
	}
	if( mode_cnt<=2)
		mode=0;
	mode=input.at<float>(0,index);

	return mode_cnt;

}

int main()   
{
	//image decelerations
	Mat showImg,curFrame,lastFrame,lastGrey,curGrey,temp_mat;
	Mat motionCondensed;
	int kernelSize=7;
	Mat flow,flowx,flowy;
	vector<Mat> flowChannels(2);
	

	VideoCapture video =VideoCapture("18_cut.avi");
	VideoWriter videoOut;
	videoOut.open("major_flow.avi", CV_FOURCC('X','V','I','D'), video.get(CV_CAP_PROP_FPS), Size(video.get(CV_CAP_PROP_FRAME_WIDTH),video.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
	if (!videoOut.isOpened())
	{
		cout  << "Could not open the output video for write! "  << endl;
		return -1;
	}
	double total_frames;
	total_frames=video.get(CV_CAP_PROP_FRAME_COUNT);
	video>>showImg;
	Mat mask_img = Mat(showImg.rows,total_frames,CV_8U);
	temp_mat=showImg.clone();
	curFrame=temp_mat(Rect(3*temp_mat.cols/4,0,temp_mat.cols/4,temp_mat.rows));
	cv::cvtColor(curFrame,lastGrey,CV_BGR2GRAY);
	//initialize the motion condensed image
	motionCondensed=Mat(showImg.rows,(int)total_frames,CV_8UC3);
	double displacementX=0, displacementY=0;
	int frameCounter=1;
	char key=0;
	while(key!=27)
	{
		video>>showImg;
		if (showImg.empty())
		{
			break;
		}
		temp_mat=showImg.clone();
		curFrame=temp_mat(Rect(3*temp_mat.cols/4,0,temp_mat.cols/4,temp_mat.rows));
		cv::cvtColor(curFrame,curGrey,CV_BGR2GRAY);
		calcOpticalFlowFarneback(lastGrey,curGrey,flow,0.5,2,16,5,5,1.1,0);
		//split(flow,flowChannels);
		//flowx=flowChannels[0];
		//flowy=flowChannels[1];
		//drawOptFlowMap(flow,curFrame,8, CV_RGB(0, 255, 0));
		//x_component(flow);
		//save_x_component(flow,mask_img(Range(frameCounter,frameCounter+1),Range::all()));
		save_x_component(flow,mask_img,frameCounter);
		drawOptFlowColumn(flow,curFrame,CUT_COLUMN,1, CV_RGB(0, 255, 0));
		imshow("optical flow",curFrame);
		waitKey(33);
		frameCounter++;

	}
	imwrite("mask.tiff",mask_img);
	waitKey();
	return 0;
}