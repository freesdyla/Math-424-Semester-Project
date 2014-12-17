#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl.h"
#include "mkl_vsl.h"

#define useMPI 1

#if useMPI
#include <mpi.h>
#endif

#include <omp.h>

using namespace std;

// image size
const int rows = 1224, cols = 1624;	
int imgSize = rows*cols;

// disparity range
const int Dmin = 1;
const int Dmax = 80;
const int Drange = Dmax - Dmin + 1;

// radius of the square window for disimilarity evaluation
const int winRadius = 9;

// stereo image pair name
const char leftImgName[] = "l.pgm";
const char rightImgName[] = "r.pgm";	

// number of threads for OpenMP
int numThreads = 8;

// MPI
int numprocs, rank, namelen;

// number of trials
const int nt = 100;
// arrays used for timing
double start[nt], end[nt];
double input_start[nt], input_end[nt];	
double output_start[nt], output_end[nt];
double random_start[nt], random_end[nt];
double main_start[nt], main_end[nt];

// prototypes
int readPGM(const char* imgName, char* imgPtr);
int writePGM(const char* imgName, char* imgPtr);
int imgCharToFloat(char* imgCharPtr, float* imgFloatPtr, int size, bool reverse);
void PatchMatchStereo(float* lImgPtr, float* rImgPtr, float* Disp_l, float* Disp_r, float* plane_l, float* plane_r, int numRows); 
float evaluateCost_p(float* base_f, float* match_f, int x, int y, int x_begin_idx, int x_end_idx,
		     float disp, float nx, float ny, float nz, int winRadius); 
void timingStat(double* start, double* end, int nt, double* average, double* sd);

int main(int argc, char *argv[])
{

#if useMPI
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
	int remainderRows = rows % numprocs;

	int bufsize, r_offset, extraRow;

	// distribute remainder rows to other procs
	if(rank < remainderRows)
	{
		r_offset = rank;
		extraRow = 1;
	}
	else
	{
		r_offset = remainderRows;
		extraRow = 0;
	}

	if( rank == 0 || rank == numprocs -1)
		bufsize = (rows/numprocs + extraRow + winRadius)*cols;
	else
		bufsize =(rows/numprocs + extraRow + 2*winRadius)*cols;
	

	// allocate image memory 
	char* lImgSegPtr_8u = new char[bufsize];
	char* rImgSegPtr_8u = new char[bufsize];

	float* lImgSegPtr_f = new float[bufsize];
	float* rImgSegPtr_f = new float[bufsize];


	// allocate disparity maps	
	float* rDispPtr = new float[bufsize];
	float* lDispPtr = new float[bufsize];

	float* rPlanePtr = new float[3*bufsize];
	float* lPlanePtr = new float[3*bufsize];
	
	MPI_File fh;
	
	for(int t=0; t<=nt; t++)
	{
	if(t>0)
		start[t-1] = MPI_Wtime();
	

	// calculate read offset
	int offset;
	if(rank == 0)
		offset = 17;
	else
		offset = (rows/numprocs)*cols*rank + r_offset*cols - winRadius*cols + 17;

	MPI_Barrier(MPI_COMM_WORLD);

	if(t>0)
		input_start[t-1] = MPI_Wtime();

	MPI_File_open(MPI_COMM_WORLD, rightImgName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

	MPI_File_read_at(fh, offset, rImgSegPtr_8u, bufsize, MPI_CHAR, &status);

	MPI_File_open(MPI_COMM_WORLD, leftImgName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

	MPI_File_read_at(fh, offset, lImgSegPtr_8u, bufsize, MPI_CHAR, &status);

	MPI_Barrier(MPI_COMM_WORLD);

	if(t>0)
		input_end[t-1] = MPI_Wtime();

	// convert unsigned char to float and normalize to [0,1]
	imgCharToFloat(lImgSegPtr_8u, lImgSegPtr_f, bufsize, false);
	imgCharToFloat(rImgSegPtr_8u, rImgSegPtr_f, bufsize, false);

	MPI_Barrier(MPI_COMM_WORLD);

	if(t>0)
		main_start[t-1] = MPI_Wtime();

	PatchMatchStereo(lImgSegPtr_f, rImgSegPtr_f, lDispPtr, rDispPtr, lPlanePtr,rPlanePtr, rows/numprocs+extraRow);
	
	MPI_Barrier(MPI_COMM_WORLD);

	if(t>0)
		main_end[t-1] = MPI_Wtime();

	//convert float disparity to char
	imgCharToFloat(lImgSegPtr_8u, rDispPtr, bufsize, true);


	MPI_Barrier(MPI_COMM_WORLD);

	if(t>0)
		output_start[t-1] = MPI_Wtime();	

	//write
	MPI_File_open(MPI_COMM_WORLD, "disp_mpi_.pgm", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

	if( rank == 0)
		MPI_File_write_at(fh, 0, lImgSegPtr_8u,	bufsize-winRadius*cols, MPI_CHAR, &status);
	else if( rank == numprocs -1 )
		MPI_File_write_at(fh, (rows/numprocs)*rank*cols + r_offset*cols, lImgSegPtr_8u+winRadius*cols, 
				bufsize-winRadius*cols, MPI_CHAR, &status);
        else
		MPI_File_write_at(fh, (rows/numprocs)*rank*cols+ r_offset*cols, lImgSegPtr_8u+winRadius*cols, 
				bufsize-2*winRadius*cols, MPI_CHAR, &status);
       
	MPI_Barrier(MPI_COMM_WORLD);

	if(t>0)
		output_end[t-1] = MPI_Wtime();

	MPI_File_close(&fh);

	MPI_Barrier(MPI_COMM_WORLD);
	
	if(t>0)
		end[t-1] = MPI_Wtime();
	}

	if(rank == 0)	
	{
	double mean = 0.0, sd = 0.0;

	cout<<"Results from rank "<<rank<<endl;	

	timingStat(input_start, input_end, nt, &mean, &sd);

	cout<<"Input: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(random_start, random_end, nt, &mean, &sd);

	cout<<"Random disp init: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(main_start, main_end, nt, &mean, &sd);

	cout<<"Main loop: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(output_start, output_end, nt, &mean, &sd);

	cout<<"Output: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(start, end, nt, &mean, &sd);
	
	cout<<"Total Time: "<<mean<<"s  SD: "<<sd<<"s"<<endl; 
	}
	
	delete[] lImgSegPtr_8u;
	delete[] rImgSegPtr_8u;
	delete[] lImgSegPtr_f;
	delete[] rImgSegPtr_f;
	delete[] lDispPtr;
	delete[] rDispPtr;
	delete[] lPlanePtr;
	delete[] rPlanePtr;

  	MPI_Finalize();
	return 0;

#endif

	
	for(int tr=0; tr<=nt; tr++)
	{
	if(tr>0)
		start[tr-1] = omp_get_wtime();

	if(tr>0)
		input_start[tr-1] = omp_get_wtime();

	// allocate left image (grayscale)
	char* lImgPtr_8u = new char[imgSize];
		
	if(readPGM(leftImgName, lImgPtr_8u) < 0)
	{
		cout<<"read left image fail"<<endl;
		delete[] lImgPtr_8u;
	}

	// allocate right image
	char* rImgPtr_8u = new char[imgSize];

	if(readPGM(rightImgName, rImgPtr_8u) < 0)
	{
		cout<<"read right image fail"<<endl;
		delete[] rImgPtr_8u;
		return -1;
	}

	if(tr>0)
		input_end[tr-1] = omp_get_wtime();

	// convert image type from char to float
	float* lImgPtr_f = new float[imgSize];
	
	imgCharToFloat(lImgPtr_8u, lImgPtr_f, imgSize, false);

	float* rImgPtr_f = new float[imgSize];

	imgCharToFloat(rImgPtr_8u, rImgPtr_f, imgSize, false);


	// PatchMatch stereo matching
	// plane and disparity random initialization
 	float* plane_r = new float[3*imgSize];
	float* plane_l = new float[3*imgSize];
	float* disp_r = new float[imgSize];
	float* disp_l = new float[imgSize];

	if(tr>0)
		random_start[tr-1] = omp_get_wtime();

	VSLStreamStatePtr stream;

	vslNewStream(&stream, VSL_BRNG_MT19937, 777);
	
 	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, imgSize, disp_r, (float)Dmin, (float)Dmax);
	
	if(tr>0)	
		random_end[tr-1] = omp_get_wtime();

// 	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, imgSize, disp_l, (float)Dmin, (float)Dmax);
	
	// generate uniformally distributed plane normal for each pixel
	for(int i=0; i<imgSize; i++)
	{
		float x1, x2;
		float nx, ny, nz;
		
		while(true)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &x1, -1.0f, 1.0f);
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &x2, -1.0f, 1.0f);
		
			if( x1*x1 + x2*x2 < 1.0f)
				break;
		}

		nx = 2*x1*sqrt(1-x1*x1-x2*x2);		
		ny = 2*x1*sqrt(1-x1*x1-x2*x2);
		nz = 1 - 2*(x1*x1 + x2*x2);

		plane_r[3*i] = nx;
		plane_r[3*i+1] = ny;
		plane_r[3*i+2] = nz;
	}		

	if(tr>0)
		main_start[tr-1] = omp_get_wtime();	
	
	// set up OpenMP
	omp_set_dynamic(false);
	omp_set_num_threads(numThreads);

	// PatchMatch main loop
	// iteration
	for(int it=0; it<1; it++)
	{
		int delta, x_begin_idx, x_end_idx, y_begin_idx, y_end_idx;		
	
		//if(it%2 == 0)
		{
			//image top left corner -> bottom right
			delta = 1;
			y_begin_idx = winRadius + 1;
			y_end_idx = rows - winRadius;
			x_begin_idx = winRadius + 1;
			x_end_idx = cols - winRadius - 1;
			
			#pragma omp parallel 
			#pragma omp for schedule(static)

			for(int y = y_begin_idx; y < y_end_idx; y += delta)
			{
				float cost;
				float min_cost_l, min_cost_r;	
				
				int vn_idx = (y-delta)*cols;
				//first element pointer of vertical neighbor of current disparity
				float* vrnebor_dptr = &disp_r[vn_idx];
				float* vlnebor_dptr = &disp_l[vn_idx];
				
				//first element pointer of vertical neighbor of current plane
				float* vrnebor_pptr = &plane_r[vn_idx*3];
				float* vlnebor_pptr = &plane_l[vn_idx*3];

				for(int x = x_begin_idx; x != x_end_idx; x += delta)
				{
					
					// check current disprity cost (right) 
					int idx = y*cols + x;
					min_cost_r = evaluateCost_p(&rImgPtr_f[idx], &lImgPtr_f[idx], x, y
							, x_begin_idx, x_end_idx, disp_r[idx], plane_r[3*idx]
							, plane_r[3*idx+1], plane_r[3*idx+2], winRadius);
					
					
					// check previous horizontal neighbor's parameters
					cost = evaluateCost_p(&rImgPtr_f[idx-delta], &lImgPtr_f[idx-delta], x, y
						, x_begin_idx, x_end_idx, disp_r[idx-delta], plane_r[3*(idx-delta)]
						, plane_r[3*(idx-delta)+1], plane_r[3*(idx-delta)+2], winRadius);
					
				
					if(cost < min_cost_r)
					{
						// update plane parameters
						plane_r[3*idx] = plane_r[3*(idx-delta)];
						plane_r[3*idx + 1] = plane_r[3*(idx-delta) + 1];
   						plane_r[3*idx + 2] = plane_r[3*(idx-delta) + 2];
   				
						// update disparity
						disp_r[idx] = disp_r[idx - delta];

						// update min cost
						min_cost_r = cost;
					}

					// check previous vertical neighbor's parameters
					cost = evaluateCost_p(&rImgPtr_f[idx-delta*cols], &lImgPtr_f[idx-delta*cols], x, y
						, x_begin_idx, x_end_idx, vrnebor_dptr[x], vrnebor_pptr[3*x]
						, vrnebor_pptr[3*x+1], vrnebor_pptr[3*x+2], winRadius);
					
					if(cost < min_cost_r)
					{
						plane_r[3*idx] = vrnebor_pptr[3*x];
						plane_r[3*idx+1] = vrnebor_pptr[3*x+1];
						plane_r[3*idx+2] = vrnebor_pptr[3*x+2];
						disp_r[idx] = vrnebor_dptr[x];
						min_cost_r = cost;
					}

#if 1 
					//plane refinement
					float nz = 1.0f, nx = 0.0f, ny = 0.0f, dr = 0.0f, cnt= 1.0f;
					
					// base disparity (right image)
					float current_d = disp_r[idx];

					//exponentially reduce disparity search range
//					for(float d = (float)Dmax; d >= 0.1f; d /= 0.5f)
					{
						//check if index out of bound
						if( current_d + x > cols - winRadius - 1 )
							current_d = 0;

						//disparity
						vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &dr, Dmin, Dmax);
						
						float norm;
						
						while(true)
						{
							// change
							float x1, y1, z1;
							vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &x1, -1.0f, 1.0f);
							vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &y1, -1.0f, 1.0f);
							vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &z1, -1.0f, 1.0f);
							
							// base place (right)
							nx = x1/cnt + plane_r[3*idx];
							ny = y1/cnt + plane_r[3*idx+1];
							nz = z1/cnt + plane_r[3*idx+2];

							//normalize
							norm = sqrt(nx*nx + ny*ny + nz*nz);						
							if( (norm!=0.0f) && (norm==norm) && (nz!=0))
								break;	
						}	
	
						nx /= norm;
						ny /= norm;
						nz /= norm;
							
						dr += current_d;

						cost = evaluateCost_p(&rImgPtr_f[idx], &lImgPtr_f[idx], x, y
					        	, x_begin_idx, x_end_idx, dr, nx, ny, nz, winRadius);

						if(cost < min_cost_r)
						{
							plane_r[3*idx] = nx;
							plane_r[3*idx+1] = ny;
							plane_r[3*idx+2] = nz;

							disp_r[idx] = dr;
							min_cost_r = cost;	
						}
						
						cnt *= 2.0f;
					}
#endif
				}
			} 
		}
	}	
	
	if(tr>0)
		main_end[tr-1] = omp_get_wtime();
	
	vslDeleteStream(&stream);

	// convert disparity map to .pgm image format
	imgCharToFloat(lImgPtr_8u, disp_r, imgSize, true);

	if(tr>0)
		output_start[tr-1] = omp_get_wtime();
	// write disparity map of the right image
	writePGM("disp.pgm", lImgPtr_8u);
	
	if(tr>0)
		output_end[tr-1] = omp_get_wtime();

	if(tr>0)
		end[tr-1] = omp_get_wtime();
	
	// delete pointers
	delete[] lImgPtr_8u;
	delete[] rImgPtr_8u;
	delete[] lImgPtr_f;
	delete[] rImgPtr_f;
	delete[] plane_r;
	delete[] plane_l;
	delete[] disp_r;
	delete[] disp_l;

	}

	double mean = 0.0, sd = 0.0;

	timingStat(input_start, input_end, nt, &mean, &sd);

	cout<<"Input: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(random_start, random_end, nt, &mean, &sd);

	cout<<"Random disp init: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(main_start, main_end, nt, &mean, &sd);

	cout<<"Main loop: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(output_start, output_end, nt, &mean, &sd);

	cout<<"Output: "<<mean*1000.0<<"ms  SD: "<<sd*1000.0<<"ms"<<endl;

	timingStat(start, end, nt, &mean, &sd);
	
	cout<<"Total Time: "<<mean<<"s  SD: "<<sd<<"s"<<endl; 

	return 0;	

}

void timingStat(double* start, double* end, int nt, double* average, double* sd)
{
	*average = 0.0;

	for(int i=0; i<nt; i++)
		*average += end[i] - start[i];

	*average /= (double)nt;

	*sd = 0.0;

	for(int i=0; i<nt; i++)
		*sd += pow(end[i] - start[i] - *average, 2);

	*sd = sqrt(*sd/(double)(nt-1));

	return;
}

int readPGM(const char* imgName, char* imgPtr)   
{
	FILE *filePtr;
	const int MAXLENGTH = 50;

	// input line for header lines
	// 1st line: P5
	// 2nd line(image size): 1624 1224
	// 3rdline(max pixel value): 255 
	// the rest are binary image data
	char line[MAXLENGTH];
	
	// open file
	if( (filePtr = fopen(imgName, "rb")) == NULL )
	{
		cout<<"Can not open"<<endl;
		fclose(filePtr);
		return -1;
	}

	// read first line
	fgets(line, MAXLENGTH, filePtr);
	
	if(line[0] != 'P' || line[1] != '5')
	{
		cout<<"Not P5 pgm format";
		fclose(filePtr);
		return -1;
	}
	
	// image size
	fgets(line, MAXLENGTH, filePtr);
	
	// max pixel value
	fgets(line, MAXLENGTH, filePtr);
	
//	offset 17
	for (int i = 0; i < rows; i++)
	{
		fread(&imgPtr[i*cols], sizeof(char), cols, filePtr); 
		if (feof(filePtr)) break;
	}

	fclose(filePtr);

	return 0;
}

int writePGM(const char* imgName, char* imgPtr)
{
	ofstream f(imgName, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
	
	// image size 
	const char widthStr[] = "1624";
	const char heightStr[] = "1224";

	f << "P5\n" << widthStr << " " << heightStr << "\n255";

	for(int i=0; i<rows; i++)
		f.write(reinterpret_cast<const char*>(&imgPtr[i*cols]), cols);

	return 0;
}

// convert char image to float image and normalize to [0,1]
// if reverse is true, convert float to char
int imgCharToFloat(char* imgCharPtr, float* imgFloatPtr, int size, bool reverse)
{
	if(!reverse)
	{
	//	#pragma omp parallel for
		for(int i=0; i<size; i++)
		{	//char to unsigned char then to float, otherwise negative
			imgFloatPtr[i] = (float)((unsigned char)imgCharPtr[i])/255.0f;
		}
	}
	else
	{
	//	#pragma omp parallel for
		for(int i=0; i<size; i++)
			imgCharPtr[i] = (char)imgFloatPtr[i]; 
	}
	return 0;
}

// PatchMatch stereo matching for MPI
void PatchMatchStereo(float* lImgPtr, float* rImgPtr, float* disp_l, float* disp_r, 
		      float* plane_l, float* plane_r, int numRows)
{
	int size;
	if( rank == 0 || rank == numprocs -1 )
		size = (numRows + winRadius)*cols;
	else
		size = (numRows + 2*winRadius)*cols;

	// plane and disparity random initialization
	VSLStreamStatePtr stream;

	vslNewStream(&stream, VSL_BRNG_MT19937, 777);
	
 	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, size, disp_r, (float)Dmin, (float)Dmax);
 //	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, size, disp_l, (float)Dmin, (float)Dmax);
	

	// generate uniformally distributed plane normal for each pixel
	for(int i=0; i<size; i++)
	{
		float x1, x2;
		float nx, ny, nz;
		
		while(true)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &x1, -1.0f, 1.0f);
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &x2, -1.0f, 1.0f);
		
			if( x1*x1 + x2*x2 < 1.0f)
				break;
		}

		nx = 2*x1*sqrt(1-x1*x1-x2*x2);		
		ny = 2*x1*sqrt(1-x1*x1-x2*x2);
		nz = 1 - 2*(x1*x1 + x2*x2);

		plane_r[3*i] = nx;
		plane_r[3*i+1] = ny;
		plane_r[3*i+2] = nz;
	}		
	
	// PatchMatch main loop
	// iteration
	for(int it=0; it<1; it++)
	{
		int delta, x_begin_idx, x_end_idx, y_begin_idx, y_end_idx;		
	
		if(it%2 == 0)
		{
			//image top left corner -> bottom right
			delta = 1;
			y_begin_idx = winRadius;
			if(rank != numprocs-1)
				y_end_idx = numRows + winRadius;
			else
				y_end_idx = numRows;

			x_begin_idx = winRadius;
			x_end_idx = cols - winRadius;

			for(int y = y_begin_idx; y < y_end_idx; y += delta)
			{
				float cost;
				float min_cost_l, min_cost_r;	
				
				int vn_idx = (y-delta)*cols;
				//first element pointer of vertical neighbor of current disparity
				float* vrnebor_dptr = &disp_r[vn_idx];
				float* vlnebor_dptr = &disp_l[vn_idx];
				
				//first element pointer of vertical neighbor of current plane
				float* vrnebor_pptr = &plane_r[vn_idx*3];
				float* vlnebor_pptr = &plane_l[vn_idx*3];

				for(int x = x_begin_idx; x != x_end_idx; x += delta)
				{
					
					// check current disprity cost (right) 
					int idx = y*cols + x;
					min_cost_r = evaluateCost_p(&rImgPtr[idx], &lImgPtr[idx], x, y
							, x_begin_idx, x_end_idx, disp_r[idx], plane_r[3*idx]
							, plane_r[3*idx+1], plane_r[3*idx+2], winRadius);
					
					
					// check previous horizontal neighbor's parameters
					cost = evaluateCost_p(&rImgPtr[idx-delta], &lImgPtr[idx-delta], x, y
						, x_begin_idx, x_end_idx, disp_r[idx-delta], plane_r[3*(idx-delta)]
						, plane_r[3*(idx-delta)+1], plane_r[3*(idx-delta)+2], winRadius);
					
				
					if(cost < min_cost_r)
					{
						// update plane parameters
						plane_r[3*idx] = plane_r[3*(idx-delta)];
						plane_r[3*idx + 1] = plane_r[3*(idx-delta) + 1];
   						plane_r[3*idx + 2] = plane_r[3*(idx-delta) + 2];
   				
						// update disparity
						disp_r[idx] = disp_r[idx - delta];

						// update min cost
						min_cost_r = cost;
					}

					// check previous vertical neighbor's parameters
					cost = evaluateCost_p(&rImgPtr[idx-delta*cols], &lImgPtr[idx-delta*cols], x, y
						, x_begin_idx, x_end_idx, vrnebor_dptr[x], vrnebor_pptr[3*x]
						, vrnebor_pptr[3*x+1], vrnebor_pptr[3*x+2], winRadius);
					
					if(cost < min_cost_r)
					{
						plane_r[3*idx] = vrnebor_pptr[3*x];
						plane_r[3*idx+1] = vrnebor_pptr[3*x+1];
						plane_r[3*idx+2] = vrnebor_pptr[3*x+2];
						disp_r[idx] = vrnebor_dptr[x];
						min_cost_r = cost;
					}
#if 1 
					//plane refinement
					float nz = 1.0f, nx = 0.0f, ny = 0.0f, dr = 0.0f, cnt= 1.0f;
					
					// base disparity (right image)
					float current_d = disp_r[idx];

					//exponentially reduce disparity search range
					//for(float d = (float)Dmax; d >= 0.1f; d *= 0.5f)
					{
						//check if index out of bound
						if( current_d + x > cols - winRadius - 1 )
							current_d = 0;

						//disparity
						vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &dr, Dmin, Dmax);
						
						float norm;
						
						while(true)
						{
							// change
							float x1, y1, z1;
							vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &x1, -1.0f, 1.0f);
							vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &y1, -1.0f, 1.0f);
							vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &z1, -1.0f, 1.0f);
							
							// base place (right)
							nx = x1/cnt + plane_r[3*idx];
							ny = y1/cnt + plane_r[3*idx+1];
							nz = z1/cnt + plane_r[3*idx+2];

							//normalize
							norm = sqrt(nx*nx + ny*ny + nz*nz);						
							
							if( (norm!=0.0f) && (norm==norm) && (nz!=0))
								break;	
						}	
	
						nx /= norm;
						ny /= norm;
						nz /= norm;
							
						dr += current_d;

						cost = evaluateCost_p(&rImgPtr[idx], &lImgPtr[idx], x, y
					        	, x_begin_idx, x_end_idx, dr, nx, ny, nz, winRadius);

						if(cost < min_cost_r)
						{
							plane_r[3*idx] = nx;
							plane_r[3*idx+1] = ny;
							plane_r[3*idx+2] = nz;

							disp_r[idx] = dr;
							min_cost_r = cost;	
						}
						
						cnt *= 2.0f;
				
					}
#endif
				}
			} 
		}
	}		
	vslDeleteStream(&stream);
}

// evaluate disparity cost for a pixel, disparity is positive always, x and y are base pixel coordinates. 
// base and match are center pixels, float type, range 0-1
// Used by MPI parallelization
float evaluateCost_p(float* basePtr_f, float* matchPtr_f, int x, int y, int x_begin_idx, int x_end_idx,
		     float disp, float nx, float ny, float nz, const int winRadius) 
{
	float cost = 1000.0f;
	const int bound = x_begin_idx > x_end_idx ? x_begin_idx : x_end_idx;

	// disparity index out of bound, return a large cost
	if( x + disp >= bound )
		return cost;
	
	float af, bf, cf;
	
	af = nx/nz*(-1.0f);
	bf = ny/nz*(-1.0f);
	cf = (nx*x + ny*y + nz*disp)/nz;

	float tmp = af*x + bf*y + cf;
	
	// overflow, return large cost
	if(tmp != tmp)
		return cost;	
	
	cost = 0.0f;

	float match_value;
	
	// aggregate cost for the window
	for(int h = -winRadius; h <= winRadius; h++)
	{
		float* bptr = basePtr_f + h*cols;
		float* mptr = matchPtr_f + h*cols;

		float tmp_d, tmp_match_idx;

		for(int w = -winRadius; w <= winRadius; w++)
		{
			float tmp_idx_f, ceil_idx_f;
			int floor_idx_i;

			tmp_d = af*(x+w) + bf*(y+h) + cf;

			// overflow
			if( tmp_d != tmp_d)
				return 1000.0f;

			// matching index in match image
			tmp_match_idx = (float)(x+w) + tmp_d;


			if( tmp_match_idx > bound || tmp_match_idx < winRadius)
				return 1000.0f;
			
			floor_idx_i = (int)tmp_match_idx;


		        match_value = mptr[floor_idx_i-x] + 
					( mptr[floor_idx_i+1-x] - mptr[floor_idx_i-x] )
					*( tmp_match_idx - floor(tmp_match_idx) );

			match_value -= bptr[w];

			cost += match_value>=0 ? match_value : (-1.0f)*match_value;
		}
	}
	return cost;
}

