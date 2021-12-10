#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

void ransac(std::vector<cv::Point3f>& points, cv::Vec4f &bestplane, int max_iter, float threshold)
{
    unsigned int n = points.size();

    if(n<3)
    {
        return;
    }

    cv::RNG rng;
    //double bestScore = -1.;
    int bestScore = 0;
    for(int i=0; i<max_iter; i++)
    {
        int i1=0, i2=0, i3=0;
        int score = 0;
        while(i1==i2||i1==i3||i2==i3)
        {
            i1 = rng(n);
            i2 = rng(n);
            i3 = rng(n);
        }
        const cv::Point3f& p1 = points[i1];
        const cv::Point3f& p2 = points[i2];
        const cv::Point3f& p3 = points[i3];
        
        cv::Point3f p12 = p2 - p1;
        cv::Point3f p13 = p3 - p1;
        //3d vector cross product
        cv::Point3f dp  = {p12.y*p13.z - p12.z*p13.y, p12.z*p13.x - p12.x*p13.z, p12.x*p13.y - p12.y*p13.x};
        //plane: dp[0]x+dp[1]y+dp[2]z+k=0 
        float k = -(p1.x*dp.x + p1.y*dp.y + p1.z*dp.z);
        
        for(int i=0;i<n;i++){
            double d = fabs(dp.x*points[i].x+dp.y*points[i].y+dp.z*points[i].z+k)/norm(dp);
            if(d < threshold) score += 1;
        }

        if(score > bestScore)
        {
            bestplane = cv::Vec4f(dp.x, dp.y, dp.z, k);
            bestplane *= 1./fabs(k);
            bestScore = score;
        }
    }
    
    std::cout<<bestScore<<std::endl;
    std::cout<<bestplane[0]<<" "<<bestplane[1]<<" "<<bestplane[2]<<" "<<bestplane[3]<<std::endl;
}


void fitLineRansac(const std::vector<cv::Point2f>& points,
                   cv::Vec4f &line,
                   int iterations = 1000,
                   double sigma = 1.,
                   double k_min = -7.,
                   double k_max = 7.)
{
    unsigned int n = points.size();

    if(n<2)
    {
        return;
    }

    cv::RNG rng;
    double bestScore = -1.;
    for(int k=0; k<iterations; k++)
    {
        int i1=0, i2=0;
        while(i1==i2)
        {
            i1 = rng(n);
            i2 = rng(n);
        }
        const cv::Point2f& p1 = points[i1];
        const cv::Point2f& p2 = points[i2];

        cv::Point2f dp = p2-p1;//直线的方向向量
        dp *= 1./norm(dp);
        double score = 0;

        if(dp.y/dp.x<=k_max && dp.y/dp.x>=k_min )
        {
            for(int i=0; i<n; i++)
            {
                cv::Point2f v = points[i]-p1;
                double d = v.y*dp.x - v.x*dp.y;//向量a与b叉乘/向量b的摸.||b||=1./norm(dp)
                //score += exp(-0.5*d*d/(sigma*sigma));//误差定义方式的一种
                if( fabs(d)<sigma )
                    score += 1;
            }
        }
        if(score > bestScore)
        {
            line = cv::Vec4f(dp.x, dp.y, p1.x, p1.y);
            bestScore = score;
        }
    }
}

int main()
{
    std::vector<cv::Point3f> points;
    
    cv::RNG rng((unsigned)time(NULL));
    for (int i = 0; i < 500; i+=10)
    {
        int temp = rng.uniform(-500,500);
        cv::Point3f point(i,temp,i+temp+1);
        points.emplace_back(point);
    }
    
    //add noise to the points on the plane
    for (int i = 0; i < 500; i+=10)
    {
        int y = rng.uniform(i-500,500-i);
        int z = rng.uniform(i+y+1-10,i+y+1+10);
        cv::Point3f point(i,y,z);;
        points.emplace_back(point);
    }

    //add random points
    for (int i = 0; i < 500; i+=20)
    {
        int x = rng.uniform(-500,500);
        int y = rng.uniform(-500,500);
        int z = rng.uniform(-500,500);

        cv::Point3f point(x,y,z);
        points.emplace_back(point);
    }


    //z = ax + by + c
    cv::Vec4f bestplane;

    ransac(points, bestplane, 1000, 0.1);
    std::cout<<"z=ax+by+c,estimate parameters:a="<<bestplane[0]<<",b="<<bestplane[1]<<",c="<<bestplane[2]<<std::endl;

    cv::viz::Viz3d window; //creating a Viz window
    //Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", cv::viz::WCoordinateSystem(10));
    //Displaying the 3D points in green
    window.showWidget("points", cv::viz::WCloud(points, cv::viz::Color::green()));
    //displaying plane
    cv::Point3d center_1(0.,0.,-bestplane[3]/bestplane[2]);
    cv::Vec3d normal(bestplane[0], bestplane[1], bestplane[2]);
    cv::Vec3d center_2(0,-bestplane[3]/bestplane[1],bestplane[3]/bestplane[2]); 
    window.showWidget("plane", cv::viz::WPlane(center_1,normal,center_2,cv::Size2d(500.0,500.0),cv::viz::Color::red()));
    window.spin();
}

