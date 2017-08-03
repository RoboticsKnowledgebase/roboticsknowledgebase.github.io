How to use Eigen Geometry library for c++
Essential Libraries

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Core>

This library helps you to do the following operations

    Declare Vectors, matrices, quaternions
    Perform operations like dot product, cross product, vector/matrix addition ,subtraction, multiplication
    Convert from one form to another. For instance one can convert quaternion to affine pose matrix and vice versa
    Use AngleAxis function to create rotation matrix in single line

For instance a rotation matrix homogeneous transform of PI/2 about z-axis can be written
as
Eigen::Affine3d T_rt(Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitZ()));

    Extract rotation matrix from Affine matrix using Eigen::Affine3d Mat.rotation( )
    Extract translation vector from Affine Matrix using Eigen::Affine3d Mat.translation( )
    Find inverse and transpose of a matrix using Mat.inverse( ) and Mat.transpose( )


The applications are the following

    Convert Pose to Quaternions and vice versa
    Find the relative pose transformations by just using simple 3D homogeneous transformation

Eigen::Affine3d T is a 4*4 homogeneous transform
external image jJcbt8gZ0CMCgfc-rOjFHNSR9GE_zF6ixk_Z7WbJnwBbF9dwEOK4Rle8kWltQMDRj9_e_UlF2TLCa5nFt6F81-mgL2F6_eot7T9I7evczgsb9XZgB6lJfGkBBxjy-816PXo5oGTy


3. Now all the transformations (rotation or translation) can be represented in homogeneous form as simple 4*4 matrix multiplications
4. Suppose you have a pose transform T of robot in the world and you want to find robot’s X-direction relative to the world then you can do this by
Eigen::Vector3d x_bearing= T.rotation * Eigen::Vector3d::UnitX();
This is an important library in c++ which gives capabilities equal to python for vectors and matrices. More helpful functions and examples can be found at the following link
__http://eigen.tuxfamily.org/dox/__
__https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html__
__https://eigen.tuxfamily.org/dox/classEigen_1_1Transform.html__
