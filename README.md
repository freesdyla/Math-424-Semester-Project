Math-424-Semester-Project
=========================

Source code for parallel PatchMatch Stereo matching using MPI, OpenMP and Cuda. 

stereo.cpp contains MPI and OpenMP version. And stereo.cu is the Cuda version.

To run the program, you need to have the two images (l.pgm & r.pgm) in your working directory.

You can download the 2 images from
https://drive.google.com/a/iastate.edu/file/d/0BwmQ0lVpCBlCZWw1SG1pWEpKVEU/view?usp=sharing
and 
https://drive.google.com/a/iastate.edu/file/d/0BwmQ0lVpCBlCVFRNeVlYUDFWdGM/view?usp=sharing

To run OpenMP version on ISU HPC-Class cluser, submit stereoOMP.script. The number of threads have to be change in stereo.cpp

To run MPI version, submit stereoMPI.script.

To run Cuda version, submit stereoCuda.script.
