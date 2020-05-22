"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 1508
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  print "Camera Intrinsics",cam_intr
  cam_poses=np.loadtxt("data/gt_poses.txt")
  print "Cam poses shape",cam_poses.shape
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread("/home/ashfaquekp/depth/%d.png"%(i+1),-1).astype(float)
    depth_im /= 5000.  # depth is saved in 16-bit PNG in millimeters
    print "Shape of depth image:",depth_im.shape
    #depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    #cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix
    cam_pose=cam_poses[3*i:3*(i+1),:]
    homo=np.array([[0.0,0.0,0.0,1.0]])
    cam_pose=np.concatenate([cam_pose,homo])
    print "Concatenated Cam pose",cam_pose
    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    if i==1:
        break
  print "Volume Bounds:",vol_bnds
  # ======================================================================================================== #
  	
