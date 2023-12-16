# Fixing gif using Neural Radiance Fields (NeRF)  
Creating a fixed gif through the application of Neural Radiance Fields (NeRF) involves achieving seamless transitions between the initial and concluding frames. The goal is to ensure that the human eye cannot discern the starting and ending points of the image sequence. In essence, the action depicted in the gif should give the illusion of perpetual repetition rather than a conventional playback experience.
  
results of a *fixed gif*:  

<p align="center">
  <img src="extra/first_scene/fixedfirstscene.gif" width="380" />
</p>

# Data 
I photographed two different objects. The first under good conditions and without any obstacles, while the second had a lot of variation in light conditions around it.
you can find all the data under ./extra folder 

# Work process

To craft the fixed gif using NeRF, follow these procedures:

1. Determine the camera position for each image using colmap. Colmap, a structure-from-motion framework, generates camera positions for a given series of images.

2. Train NeRF on the extracted images, incorporating the generated camera positions. Utilize the TensoRF framework to train NeRF specifically for the scene or object.

3. Pick the right points to form a closed circle around the object.

4. Employ the trained NeRF network on the missed points to generate new images.

5. Assemble a new gif from the newly generated images.


**0. Source gif**  

<p align="center">
  <img src="extra/first_scene/first_option/firstscene.gif" width="380" />
</p>

**1. Camera Positions**  
Colmap was employed to produce camera positions corresponding to the extracted images. The illustration below depicts the distribution of camera positions encircling the object, with each point in black denoting the frame number. The blue points trace the trajectory of the camera operator around the object, assuming the object is situated at (0,0,0). It is evident that the camera holder completed a revolution of over 360 degrees, indicating a failure to return to the initial position. Additionally, the Z-axis indicates vertical movement as the person recorded the video, capturing fluctuations in height.  

<p align="center">
  <img src="extra/first_scene/" width="380" />
</p>

**2. NeRF: Neural Radiance Fields**  

In this segment, we had the flexibility to opt for any iteration of NeRF. Our choice fell upon TensoRF, available at https://apchenstu.github.io/TensoRF/. We selected this particular version due to its lucid PyTorch implementation and brief runtime. The training Peak Signal-to-Noise Ratio (PSNR) achieved an impressive 37.451, indicating high-quality images with minimal noise around the object.

Additionally, we conducted a run using Instant-ngp, yielding favorable results. It certainly provides an alternative approach worth considering.

<p align="center">
  <img src="extra/first_scene/camerapositions.jpeg" width="380" />
</p>

<p align="center">
  <img src="extra/first_scene/selectedcircle.jpeg" width="380" />
</p>


**3. Find a perfect closed circle**  

TODO

<p align="center">
  <img src="extra/first_scene/camerapoints.jpeg" width="380" />
</p>

**4. Generate new images**  

TODO

**5. New fixed gif**  

<p align="center">
  <img src="extra/first_scene/fixedfirstscene.gif" width="380" />
</p>

