# Blur_segmentation


The presented work addresses a blur segmentation method for scene reconstruction from integral images. It focuses on using elemental images, captured from different perspectives, to reconstruct a three-dimensional image and segment different depth layers based on blur. The primary objective is to extract relevant information from each plane of interest, minimizing the effects of occlusions between planes.

In the development of the study, a 3D scene is reconstructed from a set of 2D images using a shift-and-sum method, adjusting specific parameters for each elemental image. Subsequently, blur operators such as the Laplacian variation and local binary pattern are applied to classify pixels based on their focus. The combination of these operators, along with preprocessing and postprocessing techniques, facilitates the precise segmentation of focused regions in each plane.

Finally, the obtained results are validated by comparing them with a manual reference segmentation. It is observed that the proposed method effectively minimizes noise and false classifications of out-of-focus pixels. The work concludes by highlighting the method's effectiveness in segmenting objects at different depths in complex scenes, providing a useful tool for the analysis of 3D images in various scientific and technological applications.
