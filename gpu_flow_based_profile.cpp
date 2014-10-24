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
#include <gpu/gpu.hpp>
using namespace cv;
using namespace std;

#define N_FRAMES 50
//#define GPU
# define M_PI           3.14159265358979323846  /* pi */
# define CUT_COLUMN 45
// Some defines
#define NMAX_CHARACTERS 500
#define SAVE_RESULTS 1  // Flag for saving video results

//global values and parameters
const float alpha_ = 0.12;
const float gamma_ = 5;
const float scale_factor_ = 0.9;
const int inner_iterations_ = 3;
const int outer_iterations_ = 50;
const int solver_iterations_ = 20;
const bool resize_img = false;
const float rfactor = 2.0;

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
			if (val==max_val)
			{
				cout<<"blah"<<endl;
			}
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
			//assign the value from the lookup table (i.e. LUT[mag]) to the colorImage
// 			if (mag>=1)
// 			{
// 				cout<<"looky";
// 			}
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

//******************************************************************************
//******************************************************************************

/** This function calculates rgb values from hsv color space                     */
void hsv2rgb(float h, float s, float v, unsigned char &r, unsigned char &g, unsigned char &b)
{
	float c = 0.0, hp = 0.0, hpmod2 = 0.0, x = 0.0;
	float m = 0.0, r1 = 0.0, g1 = 0.0, b1 = 0.0;

	if(h > 360)
	{
		h = h - 360;
	}

	c = v*s;   // chroma
	hp = h / 60;
	hpmod2 = hp - (float)((int)(hp/2))*2;

	x = c*(1 - fabs(hpmod2 - 1));
	m = v - c;

	if(0 <= hp && hp < 1)
	{
		r1 = c;
		g1 = x;
		b1 = 0;
	}
	else if(1 <= hp && hp < 2)
	{
		r1 = x;
		g1 = c;
		b1 = 0;
	}
	else if(2 <= hp && hp < 3)
	{
		r1 = 0;
		g1 = c;
		b1 = x;
	}
	else if(3 <= hp && hp < 4)
	{
		r1 = 0;
		g1 = x;
		b1 = c;
	}
	else if(4 <= hp && hp < 5)
	{
		r1 = x;
		g1 = 0;
		b1 = c;
	}
	else
	{
		r1 = c;
		g1 = 0;
		b1 = x;
	}

	r = (unsigned char)(255*(r1 + m));
	g = (unsigned char)(255*(g1 + m));
	b = (unsigned char)(255*(b1 + m));
}

//******************************************************************************
//******************************************************************************

/** This function draws a vector field based on horizontal and vertical flow fields   */
void drawMotionField(Mat &imgU, Mat &imgV, Mat &imgMotion,
	int xSpace, int ySpace, float cutoff, float multiplier, CvScalar color)
{
	int x = 0, y = 0;
	float *ptri;
	float deltaX = 0.0, deltaY = 0.0, angle = 0.0, hyp = 0.0;
	Point p0, p1;

	for( y = ySpace; y < imgU.rows; y += ySpace )
	{
		for(x = xSpace; x < imgU.cols; x += xSpace )
		{
			p0.x = x;
			p0.y = y;

			ptri = imgU.ptr<float>(y);
			deltaX = ptri[x];

			ptri = imgV.ptr<float>(y);
			deltaY = ptri[x];

			angle = atan2(deltaY, deltaX);
			hyp = sqrt(deltaX*deltaX + deltaY*deltaY);

			if(hyp > cutoff)
			{
				p1.x = p0.x + cvRound(multiplier*hyp*cos(angle));
				p1.y = p0.y + cvRound(multiplier*hyp*sin(angle));

				cv::line(imgMotion,p0,p1,color,1,CV_AA,0);

				p0.x = p1.x + cvRound(2*cos(angle-M_PI + M_PI/4));
				p0.y = p1.y + cvRound(2*sin(angle-M_PI + M_PI/4));
				cv::line( imgMotion, p0, p1, color,1, CV_AA, 0);

				p0.x = p1.x + cvRound(2*cos(angle-M_PI - M_PI/4));
				p0.y = p1.y + cvRound(2*sin(angle-M_PI - M_PI/4));
				cv::line( imgMotion, p0, p1, color,1, CV_AA, 0);
			}
		}
	}
}

//******************************************************************************
//******************************************************************************

/** Draws the circular legend for the color field, indicating direction and magnitude */
void drawLegendHSV(Mat &imgColor, int radius, int px, int py)
{
	unsigned char *legend_ptr, *img_ptr;
	float angle = 0.0, h = 0.0, s = 0.0, v = 0.0, legend_max_s = 0.0;
	unsigned char r = 0, g = 0, b = 0;
	int deltaX = 0, deltaY = 0, mod = 0;
	int width = radius*2 + 1;
	int height = width;

	Mat imgLegend = cv::Mat::zeros(Size(width,height),CV_8UC3);

	legend_max_s = radius*sqrt((float)2);

	for( int y = 0; y < imgLegend.rows; y++ )
	{
		legend_ptr = imgLegend.ptr<unsigned char>(y);

		for( int x = 0; x < imgLegend.cols; x++ )
		{
			deltaX = x-radius;
			deltaY = -(y-radius);
			angle = atan2((float)deltaY,deltaX);

			if( angle < 0.0 )
			{
				angle += 2*M_PI;
			}

			h = angle * 180.0 / M_PI;
			s = sqrt((float)deltaX*deltaX + deltaY*deltaY) / legend_max_s;
			v = 0.9;

			hsv2rgb(h,s,v,r,g,b);

			legend_ptr[3*x] = b;
			legend_ptr[3*x+1] = g;
			legend_ptr[3*x+2] = r;

			mod = (x-radius)*(x-radius) + (y-radius)*(y-radius);

			if(  mod < radius*radius )
			{
				img_ptr = imgColor.ptr<unsigned char>(py+y);
				img_ptr[3*(px+x)] = legend_ptr[3*x];
				img_ptr[3*(px+x)+1] = legend_ptr[3*x+1];
				img_ptr[3*(px+x)+2] = legend_ptr[3*x+2];
			}
		}
	}

	// Draw a black circle over the legend
	cv::circle(imgColor,cv::Point(px+radius,py+radius),radius,CV_RGB(0,0,0),2,8,0);
}

//******************************************************************************
//******************************************************************************

/** This function draws a color field representation of the flow field          */
void drawColorField(Mat &imgU, Mat &imgV, Mat &imgColor)
{
	Mat imgColorHSV = cv::Mat::zeros(Size(imgColor.cols,imgColor.rows),CV_32FC3);

	float max_s = 0;
	float *hsv_ptr, *u_ptr, *v_ptr;
	unsigned char *color_ptr;
	unsigned char r = 0, g = 0, b = 0;
	float angle = 0.0;
	float h = 0.0, s = 0.0, v = 0.0;
	float deltaX = 0.0, deltaY = 0.0;
	int x = 0, y = 0;

	// Generate hsv image
	for( y = 0; y < imgColor.rows; y++ )
	{
		hsv_ptr = imgColorHSV.ptr<float>(y);
		u_ptr = imgU.ptr<float>(y);
		v_ptr = imgV.ptr<float>(y);

		for( x = 0; x < imgColorHSV.cols; x++)
		{
			deltaX = u_ptr[x];
			deltaY = -v_ptr[x];
			angle = atan2(deltaY,deltaX);

			if( angle < 0)
			{
				angle += 2*M_PI;
			}

			hsv_ptr[3*x] = angle * 180 / M_PI;
			hsv_ptr[3*x+1] = sqrt(deltaX*deltaX + deltaY*deltaY);
			hsv_ptr[3*x+2] = 0.9;

			if( hsv_ptr[3*x+1] > max_s )
			{
				max_s = hsv_ptr[3*x+1];
			}
		}
	}

	// Generate color image
	for(y = 0; y < imgColor.rows; y++ )
	{
		hsv_ptr = imgColorHSV.ptr<float>(y);
		color_ptr = imgColor.ptr<unsigned char>(y);

		for( x = 0; x < imgColor.cols; x++)
		{
			h = hsv_ptr[3*x];
			s = hsv_ptr[3*x+1] / max_s;
			v = hsv_ptr[3*x+2];

			hsv2rgb(h,s,v,r,g,b);

			color_ptr[3*x] = b;
			color_ptr[3*x+1] = g;
			color_ptr[3*x+2] = r;
		}
	}

	drawLegendHSV(imgColor,15,25,15);
}

int main()   
{
	// Variables for CUDA Brox Optical flow
	gpu::GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
	Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1;
	Mat frame0_32, frame1_32, imgU, imgV;
	Mat motion_flow, flow_rgb;
	int nframes = 0, width = 0, height = 0;
	char cad[NMAX_CHARACTERS];

	// Variables for measuring computation times
	double t1 = 0.0, t2 = 0.0, tdflow = 0.0, tvis = 0.0;

	// Show CUDA information
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	// Create OpenCV windows
	namedWindow("Dense Flow",CV_WINDOW_OPENGL);
	namedWindow("Motion Flow",CV_WINDOW_NORMAL);

	// Create the optical flow object
	cv::gpu::BroxOpticalFlow dflow(alpha_,gamma_,scale_factor_,inner_iterations_,outer_iterations_,solver_iterations_);

	// Open the video file
	VideoCapture cap =VideoCapture("16_cut_small.avi");

	if( cap.isOpened() == 0 )
	{
		return -1;
	}
	cap >> frame1_rgb_;

	if( resize_img == true )
	{
		frame1_rgb = cv::Mat(Size(cvRound(frame1_rgb_.cols/rfactor),cvRound(frame1_rgb_.rows/rfactor)),CV_8UC3);
		width = frame1_rgb.cols;
		height = frame1_rgb.rows;
		cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width,height),0,0,INTER_LINEAR);
	}
	else
	{
		frame1_rgb = cv::Mat(Size(frame1_rgb_.cols,frame1_rgb_.rows),CV_8UC3);
		width = frame1_rgb.cols;
		height = frame1_rgb.rows;
		frame1_rgb_.copyTo(frame1_rgb);
	}

	// Allocate memory for the images
	frame0_rgb = cv::Mat(Size(width,height),CV_8UC3);
	flow_rgb = cv::Mat(Size(width,height),CV_8UC3);
	motion_flow = cv::Mat(Size(width,height),CV_8UC3);
	frame0 = cv::Mat(Size(width,height),CV_8UC1);
	frame1 = cv::Mat(Size(width,height),CV_8UC1);
	frame0_32 = cv::Mat(Size(width,height),CV_32FC1);
	frame1_32 = cv::Mat(Size(width,height),CV_32FC1);

	// Convert the image to grey and float
	cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
	frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);

	while( frame1.empty() == false )
	{
		if( nframes >= 1 )
		{
			// Upload images to the GPU
			frame1GPU.upload(frame1_32);
			frame0GPU.upload(frame0_32);

			// Do the dense optical flow
			dflow(frame0GPU,frame1GPU,uGPU,vGPU);

			uGPU.download(imgU);
			vGPU.download(imgV);

		}

		if( nframes >= 1 )
		{

			// Draw the optical flow results
			drawColorField(imgU,imgV,flow_rgb);

			frame1_rgb.copyTo(motion_flow);
			drawMotionField(imgU,imgV,motion_flow,15,15,.0,3,CV_RGB(0,255,0));

			// Visualization
			imshow("Dense Flow",flow_rgb);
			imshow("Motion Flow",motion_flow);

			waitKey(3);

		}

		// Save results
		if( SAVE_RESULTS == true )
		{
			sprintf(cad,"./output/motion/image%04d.jpg",nframes);
			imwrite(cad,motion_flow);

			sprintf(cad,"./output/flow/image%04d.jpg",nframes);
			imwrite(cad,flow_rgb);
		}

		// Set the information for the previous frame
		frame1_rgb.copyTo(frame0_rgb);
		cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
		frame0.convertTo(frame0_32,CV_32FC1,1.0/255.0,0);

		// Read the next frame
		nframes++;
		cap >> frame1_rgb_;

		if( frame1_rgb_.empty() == false )
		{
			if( resize_img == true )
			{
				cv::resize(frame1_rgb_,frame1_rgb,cv::Size(width,height),0,0,INTER_LINEAR);
			}
			else
			{
				frame1_rgb_.copyTo(frame1_rgb);
			}

			cvtColor(frame1_rgb,frame1,CV_BGR2GRAY);
			frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);
		}
		else
		{
			break;
		}

		cout << "Frame Number: " << nframes << endl;
		cout << "Time Dense Flow: " << tdflow << endl;
		cout << "Time Visualization: " << tvis << endl << endl;
	}

	// Destroy the windows
	destroyAllWindows();


	Mat FG;
	FG=imread("FG0.tiff",0);
	if (FG.empty())
	{
		cout<<"could not load foreground model!"<<endl;
		return 0;
	}
	return 0;
}