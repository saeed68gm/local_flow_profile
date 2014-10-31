#include <string>   
#include <cv.h>
#include <highgui\highgui.hpp>
#include <opencv2\opencv.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\legacy\legacy.hpp>
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
# define CUT_COLUMN 45

//Gray lookup table
uchar GrayTable[256];

#define max(a, b)  ((a > b) ? (a) : (b)) 
#define min(a, b)  ((a < b) ? (a) : (b) )

//
void buildGrayTable(uchar* GrayTable)
{
	uchar val=0;
	for (int i=0;i<256;i++)
	{
		GrayTable[i]=val;
		val+=2;
	}
}

//map the double values to a range of values
int doubleToIntMap(double input, int begin, int end)
{
	int output;
	if((int)abs(input)>end)
	{
		output=(int)abs(input);
	}
	else
		output=(int)abs(input);
	return output;
}

void find_mat_max(Mat& src, float& max_val,float& min_val)
{
	float min_found=FLT_MAX;
	float max_found=-FLT_MAX;
	for (int row=0;row<src.rows;row++)
	{
		for (int col=0;col<src.cols;col++)
		{
			//find min and max
			float val=src.at<float>(row,col);
			max_found=max(max_val,val);
			min_found=min(min_val	,val);
		}
	}
	min_val=min_found;
	max_val=max_found;
}

//normalized the image values to values between min and max
//simple normalization
//input and output should be of type uchar
void image_normalize(Mat& src,Mat& dst, int new_max, int new_min)
{
	uchar max_val=0;
	uchar min_val=255;
	for (int row=0;row<src.rows;row++)
	{
		for (int col=0;col<src.cols;col++)
		{
			//find min and max
			uchar val=src.at<uchar>(row,col);
			max_val=max(max_val,val);
			min_val=min(min_val	,val);
		}
	}
	double coeff=0;
	coeff=(double)(new_max-new_min)/(double)(max_val-min_val);
	for (int row=0;row<src.rows;row++)
	{
		for (int col=0;col<src.cols;col++)
		{
			//find min and max
			uchar val=src.at<uchar>(row,col);
			uchar new_val=((val-min_val)*coeff)+new_min;
			dst.at<uchar>(row,col)=new_val;
		}
	}

}

static void OptFlowMagColorVis(const Mat& flow, Mat& colorImage)
{
	for(int y = 0; y < colorImage.rows; y += 1)
		for(int x = 0; x < colorImage.cols; x += 1)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x); 
			double mag=sqrt((fxy.x*fxy.x)+(fxy.y*fxy.y));
			int index=doubleToIntMap(mag,0,256);
			auto dep=colorImage.channels();
			if (colorImage.channels()==1)
			{
				colorImage.at<uchar>(y,x)=GrayTable[index];
			}
			else
				cout<<"hahaha, I haven't write the code for a color image, you SIR are in trouble!"<<endl;
		}
}

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
//visualization for a range of columns
static void drawOptFlowColumn(const Mat& flow, Mat& cflowmap, int column_begin,int column_end,int step, const Scalar& color,double line_thickness=1)
{
	for(int y = 0; y < cflowmap.rows; y += step)
	{
		double sum=0;
		Point2f fxy;
		for (int col=column_begin;col<column_end;col++)
		{
			const Point2f& tempxy = flow.at<Point2f>(y, col);
			fxy=tempxy;
			sum+=abs(fxy.x);
		}
		int column=(column_end+column_begin)/2;
		sum=sum/(column_end-column_begin);
		line(cflowmap, Point(column,y), Point(cvRound(column+sum), cvRound(y+fxy.y)),color,line_thickness);
		//circle(cflowmap, Point(x,y), 2, color, -1);
	}
}
void x_component(Mat& input, Mat& output, int frameNo)
{
	for (int row=0;row<input.rows;row++)
	{
		output.at<uchar>(row,frameNo)=input.at<uchar>(row,CUT_COLUMN);

	}
}

void x_component_range(Mat& input, Mat& output, int frameNo)
{
	int delta=12;
	for (int row=0;row<input.rows;row++)
	{
		double sum=0;
		Point2f fxy=0;
		for (int col=-delta+CUT_COLUMN;col<+delta+CUT_COLUMN;col++)
		{
			Point2f& fxy=input.at<Point2f>(row,col);
			fxy.y=0;
			fxy.x=abs(fxy.x);
			sum +=fxy.x;
			if (sum>=DBL_MAX)
			{
				cout<<"warning, double over flow"<<endl;
			}
		}
		sum=sum/((2*delta+1));
		if (sum>=255)
		{
			cout<<"warning,uchar over flow"<<endl;
		}
		int mask_row=row/2;
		int mask_col=frameNo*2+(row%2);
		//output.at<uchar>(mask_row,mask_col)=(uchar)fxy.x;
		//output.at<uchar>(row,frameNo)=(uchar)fxy.x;
		output.at<uchar>(row,frameNo)=(uchar)sum;
		//cout<<row<<endl;
	}
}
void save_x_component(Mat& input, Mat& output, int frameCounter, Mat& FG, int cut_position)
{

	for (int row=0;row<FG.rows;row++)
	{
		for (int col=0;col<FG.cols;col++)
		{
			uchar fg_value=FG.at<uchar>(row,col);
			if (fg_value!=0)
			{
				int flow_row=0,flow_col=0;
				flow_col=col/2;
				flow_row=(row*2)+(col%2);
				Point2f& fxy=input.at<Point2f>(flow_row,cut_position);
				fxy.y=0;
				fxy.x=abs(fxy.x);
				output.at<uchar>(row,col)=(uchar)fxy.x;
			}
			else
			{
				output.at<uchar>(row,col)=0;
			}
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
//wrapper function around the c API cvCalcOpticalFlowHS
//both input images must be 8bit single channel images
void CalcOpticalFlowHS(Mat& last, Mat& current,Mat& flow, int use_pre,double lambda, CvTermCriteria& criteria )
{
	IplImage* cvLast=&(IplImage)last;
	IplImage* cvCur=&(IplImage)current;
	CvMat* fx,*fy;
	fx=cvCreateMat(last.rows,last.cols,CV_32FC1);
	fy=cvCreateMat(last.rows,last.cols,CV_32FC1);
	cvSetZero(fx);
	cvSetZero(fy);
	cvCalcOpticalFlowHS(cvLast,cvCur,use_pre,fx,fy,lambda,criteria);
	Mat flowx=fx;
	Mat flowy=fy;
	//flowx=Mat::zeros(flowy.size(),flowy.type());
	vector<Mat> arrOfMat= vector<Mat>();
	arrOfMat.push_back(flowx);
	arrOfMat.push_back(flowy);
	cv::merge(arrOfMat,flow);
	return;

}


int main()   
{
	//image decelerations
	Mat showImg,curFrame,lastFrame,lastGray,curGray,nextGray,temp_mat;
	Mat motionCondensed;
	int kernelSize=7;
	Mat grad,flowx,flowy,nextGrad,laplacian;
	vector<Mat> flowChannels(2);
	int thresh=5;
	buildGrayTable(GrayTable);
	
	//the array to keep track of local max flows
	//list<double,int> local_max_list;
	//double local_max=0;

	Mat FG;
	FG=imread("FG0.tiff",0);
	if (FG.empty())
	{
		cout<<"could not load foreground model!"<<endl;
		return 0;
	}

	VideoCapture video =VideoCapture("16_cut_small.avi");

	double total_frames;
	total_frames=video.get(CV_CAP_PROP_FRAME_COUNT);
	video>>showImg;
	Mat mask_img = Mat(FG.rows,FG.cols,CV_8U);
	mask_img.setTo(0);
	Mat mask_long=Mat(FG.rows*2,FG.cols/2,CV_8U);
	mask_long.setTo(0);
	temp_mat=showImg.clone();
	Rect roi=Rect(200,0,100,temp_mat.rows);
	//curFrame=temp_mat(Rect(3*temp_mat.cols/4,0,temp_mat.cols/4,temp_mat.rows));
	curFrame=temp_mat(roi);
	cv::cvtColor(curFrame,lastGray,CV_BGR2GRAY);
	//initialize the motion condensed image
	motionCondensed=Mat(showImg.rows,(int)total_frames,CV_8UC3);
	double displacementX=0, displacementY=0;

	VideoWriter videoOut;
	videoOut.open("major_flow.avi", CV_FOURCC('D','I','V','X'), video.get(CV_CAP_PROP_FPS),lastGray.size(), true);
	if (!videoOut.isOpened())
	{
		cout  << "Could not open the output video for write! "  << endl;
		return -1;
	}

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
		curFrame=temp_mat(roi);
		cv::cvtColor(curFrame,curGray,CV_BGR2GRAY);
		cv::absdiff(curGray,lastGray,grad);

		video>>showImg;
		temp_mat=showImg.clone();
		curFrame=temp_mat(roi);
		cv::cvtColor(curFrame,nextGray,CV_BGR2GRAY);
		cv::absdiff(nextGray,curGray,nextGrad);

		cv::absdiff(nextGrad,grad,laplacian);
		//laplacian=nextGrad-grad;

// 		float a,b;
// 		find_mat_max(flow,a,b);
		//drawOptFlowMap(flow,curFrame,8, CV_RGB(0, 255, 0));
		//OptFlowMagColorVis(flow,curGrey);
		//imshow("gradient magnitude",grad);
		imshow("laplacian ",laplacian);
		videoOut<<curGray;
		//save_x_component(flow,mask_img,frameCounter, FG,CUT_COLUMN);
		//x_component(flow,mask_long,frameCounter);
		x_component(grad,mask_long,frameCounter);
		Mat temp_mask= Mat(mask_long);
	
		image_normalize(mask_long,temp_mask,255,0);
		imwrite("mask-1.tiff",temp_mask);
		//drawOptFlowColumn(flow,curFrame,CUT_COLUMN,1, CV_RGB(0, 255, 0));
		//drawOptFlowColumn(flow,curFrame,CUT_COLUMN-12,CUT_COLUMN+12,1, CV_RGB(0, 255, 0));
		imshow("optical flow",curFrame);
		waitKey(10);
		lastGray=curGray.clone();
		frameCounter++;

	}
	Mat temp_mask;
	resize(temp_mask,mask_img,FG.size());
	bitwise_and(mask_img,FG,mask_img);
	
	imwrite("mask-long.tiff",mask_img);
	return 0;
}