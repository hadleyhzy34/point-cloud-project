#include <iostream>
#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

using namespace std::chrono_literals;

void align(                                                                     
         pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr registration,      
         const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,                
         const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,                
         pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)                      
 {                                                                               
     // Setting scale dependent NDT parameters                                   
     // Setting minimum transformation difference for termination condition.     
     registration->setTransformationEpsilon (0.01);                              
     // Setting maximum step size for More-Thuente line search.                  
     //registration->setStepSize (0.1);                                          
     //Setting Resolution of NDT grid structure (VoxelGridCovariance).           
     //registration->setResolution (1.0);                                        
                                                                                 
     // Setting max number of registration iterations.                           
     registration->setMaximumIterations (35);                                    
                                                                                 
     registration->setInputTarget(target_cloud);                                 
     registration->setInputSource(source_cloud);                                 
                                                                                 
     clock_t start,end;                                                          
     start = clock();                                                            
     registration->align(*output_cloud);                                         
     end = clock();                                                              
                                                                                 
     //transforming unfiltered, input cloud using found transform.               
     pcl::transformPointCloud (*source_cloud, *output_cloud, registration->getFinalTransformation ());
                                                                                 
     std::cout << "processing time is: " << ((double)(end - start)) / CLOCKS_PER_SEC << "[sec]" << std::endl;
                                                                                 
     std::cout << "Registration has converged:" << registration->hasConverged()  
               << " fitness score: " << registration->getFitnessScore() << std::endl;
                                                                                 
     std::cout<<registration->getFinalTransformation()<<std::endl;    
}

int
main (int argc, char** argv)
{
  // Loading first scan of room.
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("room_scan2.pcd", *input_cloud) == -1)
  {
    PCL_ERROR ("Couldn't read file room_scan2.pcd \n");
    return (-1);
  }
  std::cout << "Loaded " << input_cloud->size () << " data points from room_scan2.pcd" << std::endl;
    
  //defining a rotation matrix and translation vector
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

  // A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)    
  double theta = M_PI / 8;  // The angle of rotation in radians               
  transformation_matrix (0, 0) = std::cos (theta);
  transformation_matrix (0, 1) = -sin (theta);
  transformation_matrix (1, 0) = sin (theta);
  transformation_matrix (1, 1) = std::cos (theta);
  
  // A translation on Z axis (0.4 meters)
  transformation_matrix (0, 3) = 0.3;
  transformation_matrix (1, 3) = 0.5;
  transformation_matrix (2, 3) = -0.2;

  // Display in terminal the transformation matrix
  std::cout << "Applying this rigid transformation to: input_cloud -> ndt_cloud" << std::endl;

  // Executing the transformation                                             
  pcl::transformPointCloud (*input_cloud, *target_cloud, transformation_matrix);

  // Filtering input scan to roughly 10% of original size to increase speed of registration.
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize (0.2, 0.2, 0.2);
  approximate_voxel_filter.setInputCloud (input_cloud);
  approximate_voxel_filter.filter (*filtered_cloud);
  std::cout << "Filtered cloud contains " << filtered_cloud->size ()
            << " data points from room_scan2.pcd" << std::endl;

  // Initializing Normal Distributions Transform (NDT).
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  ndt.setTransformationEpsilon (0.01);
  // Setting maximum step size for More-Thuente line search.
  ndt.setStepSize (0.1);
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
  ndt.setResolution (1.0);

  // Setting max number of registration iterations.
  ndt.setMaximumIterations (35);
  // ndt.setMaximumIterations (10);

  // Setting point cloud to be aligned.
  ndt.setInputSource (filtered_cloud);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget (target_cloud);

  // Set initial alignment estimate found using robot odometry.
  Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
  Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

  // Calculating required rigid transform to align the input cloud to the target cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  //ndt.align (*output_cloud, init_guess);
  ndt.align(*output_cloud);

  std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
            << " score: " << ndt.getFitnessScore () << std::endl;

  // Transforming unfiltered, input cloud using found transform.
  pcl::transformPointCloud (*input_cloud, *output_cloud, ndt.getFinalTransformation ());

  // Saving transformed input cloud.
  pcl::io::savePCDFileASCII ("room_scan2_transformed.pcd", *output_cloud);
    
  //print transformation matrix
  std::cout<<ndt.getFinalTransformation() <<std::endl;

  pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  ndt_omp->setResolution(1.0);                                                  
  ndt_omp->setNumThreads(1);                                                    
  ndt_omp->setNeighborhoodSearchMethod(pclomp::KDTREE);                         
  ndt_omp->setMaximumIterations(35);
   
  ndt_omp->setInputSource(filtered_cloud);
  ndt_omp->setInputTarget(target_cloud);
  ndt_omp->align(*output_cloud);
  std::cout<<ndt_omp->getFinalTransformation()<<std::endl;

  // Initializing point cloud visualizer
  pcl::visualization::PCLVisualizer::Ptr
  viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer_final->setBackgroundColor (0, 0, 0);
    
  // Coloring and visualizing target cloud (red).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  target_color (target_cloud, 255, 0, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (target_cloud, target_color, "target cloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "target cloud");

  // Coloring and visualizing transformed filtered input cloud (green).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  output_color (output_cloud, 0, 255, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (output_cloud, output_color, "output cloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "output cloud");
  
  /*
  // Coloring and visualizing unfiltered input cloud (green).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  input_color (input_cloud, 0, 0, 255);
  viewer_final->addPointCloud<pcl::PointXYZ> (input_cloud, input_color, "input cloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "input cloud");
  */

  // Starting visualizer
  viewer_final->addCoordinateSystem (1.0, "global");
  viewer_final->initCameraParameters ();

  // Wait until visualizer window is closed.
  while (!viewer_final->wasStopped ())
  {
    viewer_final->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }

  return (0);
}
