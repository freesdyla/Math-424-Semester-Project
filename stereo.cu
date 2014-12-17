#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

using namespace std;

// image size
int rows = 1224, cols = 1624;	
int imgSize = rows*cols;

// iterations for stereo matching algorithm
int iteration = 1;

// disparity range
int Dmin = 1;
int Dmax = 80;
int Drange = Dmax - Dmin + 1;
//int winRadius = 9;

// device image pointer
float* dLImgPtr_f = NULL;
float* dRImgPtr_f = NULL;
size_t lPitch, rPitch;

// texture memory for stereo image pair <Type, Dim, ReadMode>
texture<float, 2, cudaReadModeElementType> lTex;
texture<float, 2, cudaReadModeElementType> rTex; 

// timing arrays
const int nt = 2;
double start[nt], end[nt];
double random_start[nt], random_end[nt];
double main_start[nt], main_end[nt];


// evaluate window-based disimilarity
__device__ float evaluateCost(float u, float v, float matchIdx, int cols, int rows, int winRadius)
{
	float cost = 0.0f;
	
	for(int h=-winRadius; h<=winRadius; h++)
	{
		for(int w=-winRadius; w<=winRadius; w++)
		{
			cost += fabsf(tex2D(lTex, matchIdx+ w/(float)cols, v+h/(float)rows) 
					- tex2D(rTex, u+w/(float)cols, v+h/(float)rows));	
		}
	}

	return cost;
}

// disparity pointer in device global memory
__global__ void stereoMatching(float* dRDispPtr, float* dRPlanes, int cols, int rows, curandState* states, int iteration)
{

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	int winRadius = 9;

	// does not need to process borders
	if(x>=cols-winRadius || x<winRadius || y>=rows-winRadius || y<winRadius)
		return;
	
	float u = x/(float)cols;
	float v = y/(float)rows;
	
	int idx = y*cols +x;
	
	// if 1st iteration, enforce planes to be fronto-parallel
	if(iteration != 0)
	{
		// x of a unit normal vector
		dRPlanes[idx*3] = 0.0f;
		// y 
		dRPlanes[idx*3+1] = 0.0f; 
		// z
		dRPlanes[idx*3+2] = 1.0f;
	}
	
	// evaluate disparity of current pixel
	float min_cost = 0.0f;
	float cost = 0.0f;
	float tmp_disp = dRDispPtr[idx];
	float matchIdx = u + tmp_disp*80.0f/(float)cols;

	min_cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);

	// evaluate disparity of left neighbor 
	cost = 0.0f;
	tmp_disp = dRDispPtr[idx-1];
	matchIdx = u + tmp_disp*80.0f/(float)cols;

	cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);
	// update current disparity if lower cost from neighbor's
	if(cost < min_cost)
	{
		min_cost = cost;
		dRDispPtr[idx] = tmp_disp;
	}

	// evaluate disparity of upper neighbor
	cost = 0.0f;
	tmp_disp =  dRDispPtr[idx-cols];
	matchIdx = u + tmp_disp*80.0f/(float)cols;

	cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);

	if(cost < min_cost)
	{
		min_cost = cost;
		dRDispPtr[idx] = tmp_disp;	
	}
	
	// evaluate another valid random disparitiy (within border) in case it is trapped at a local minima
	matchIdx= -1.0f;
		
	while(matchIdx <(float)winRadius/cols || matchIdx >=(float)(cols-winRadius)/cols )
	{
		tmp_disp = curand_uniform(&states[idx]);
	
		matchIdx = u + tmp_disp*80.0f/(float)cols;
	}
		
	cost = evaluateCost(u, v, matchIdx, cols, rows, winRadius);

	if(cost<min_cost)
	{
		min_cost = cost;
		dRDispPtr[idx] = tmp_disp;	
	}

	return;
}

// initialize random states
__global__ void init(unsigned int seed, curandState_t* states, int cols)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int idx = y*cols+x;
	curand_init(seed, idx, 0, &states[idx]);
} 


// read .pgm image
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
int imgCharToFloat(char* imgCharPtr, float* imgFloatPtr, bool reverse)
{
	if(!reverse)
	{
	//	#pragma omp parallel for
		for(int i=0; i<imgSize; i++)
			imgFloatPtr[i] = (float)imgCharPtr[i];///255.0f;
	}
	else
	{
	//	#pragma omp parallel for
		for(int i=0; i<imgSize; i++)
			imgCharPtr[i] = (char)(imgFloatPtr[i]*80.0f); 
	}
	return 0;
}

// for timing
struct timeval timerStart;

void StartTimer()
{
	gettimeofday(&timerStart, NULL);
}

// time elapsed in ms
double GetTimer()
{
	struct timeval timerStop, timerElapsed;
	gettimeofday(&timerStop, NULL);
	timersub(&timerStop, &timerStart, &timerElapsed);
	return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
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



int main(int argc, char** argv)
{
	const char leftImgName[] = "l.pgm";
	const char rightImgName[] = "r.pgm";

	// allocate left image (grayscale)
	char* lImgPtr_8u = new char[imgSize];
		
	if(readPGM(leftImgName, lImgPtr_8u) < 0)
	{
		cout<<"read left image fail"<<endl;
		delete[] lImgPtr_8u;
		return -1;
	}

	// allocate right image
	char* rImgPtr_8u = new char[imgSize];

	if(readPGM(rightImgName, rImgPtr_8u) < 0)
	{
		cout<<"read right image fail"<<endl;
		delete[] rImgPtr_8u;
		return -1;
	}

	// convert image type from char to float
	float* lImgPtr_f = new float[imgSize];
	
	imgCharToFloat(lImgPtr_8u, lImgPtr_f, false);

	float* rImgPtr_f = new float[imgSize];

	imgCharToFloat(rImgPtr_8u, rImgPtr_f, false);


	// allocate pitch memory on device for left and right image
	if(cudaSuccess != cudaMallocPitch(&dLImgPtr_f, &lPitch, cols*sizeof(float), rows))
		cout<<"MallocPitch left error"<<endl;

	if(cudaSuccess != cudaMallocPitch(&dRImgPtr_f, &rPitch, cols*sizeof(float), rows))
		cout<<"MallocPitch right error"<<endl;

	// allocate global memory on device for right disparity map
	float* dRDisp;
	if(cudaSuccess != cudaMalloc(&dRDisp, cols*sizeof(float)*rows))
		cout<<"Malloc disp error"<<endl;

	// allocate global memory on device for right planes
	float* dRPlanes;
	if(cudaSuccess != cudaMalloc(&dRPlanes, cols*3*sizeof(float)*rows))
		cout<<"Malloc planes error"<<endl;
	

	// copy images from host to device
	if(cudaSuccess != cudaMemcpy2D(dLImgPtr_f, lPitch, lImgPtr_f, sizeof(float)*cols, 
		sizeof(float)*cols, rows, cudaMemcpyHostToDevice))
		cout<<"Memcpy2D left error"<<endl;

	if(cudaSuccess != cudaMemcpy2D(dRImgPtr_f, rPitch, rImgPtr_f, sizeof(float)*cols, 
		sizeof(float)*cols, rows, cudaMemcpyHostToDevice))
		cout<<"Memcpy2D right error"<<endl;


	// setup texture
	lTex.addressMode[0] = cudaAddressModeClamp;
	lTex.addressMode[1] = cudaAddressModeClamp;
	lTex.filterMode = cudaFilterModeLinear;
	lTex.normalized = true;

	rTex.addressMode[0] = cudaAddressModeClamp;
	rTex.addressMode[1] = cudaAddressModeClamp;
	rTex.filterMode = cudaFilterModeLinear;
	rTex.normalized = true;
	
	// Bind linear memory  to the texture memory
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	if(cudaSuccess != cudaBindTexture2D(0, lTex, dLImgPtr_f, desc, cols, rows, lPitch))
		cout<<"Bind left tex error"<<endl;

	if(cudaSuccess != cudaBindTexture2D(0, rTex, dRImgPtr_f, desc, cols, rows, rPitch))
		cout<<"Bind right tex error"<<endl;


	// launch kernel 
	dim3 blockSize(16, 16);

	dim3 gridSize( (cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.x - 1)/blockSize.x);

	StartTimer();

	// allocate memory for states
	curandState_t* states;
	cudaMalloc(&states, imgSize*sizeof(curandState_t));
	// initialize random states
	init<<<gridSize, blockSize>>>(1234, states, cols);	
	
	cudaDeviceSynchronize();

	cout<<"Init states time: "<<GetTimer()<<"ms"<<endl;

	curandGenerator_t gen;

	for(int t=0; t<=nt; t++)
	{

	cudaDeviceSynchronize();

	if(t>0)
	{ 
		StartTimer();	
		random_start[t-1] = 0.0;
	}
	
	// host CURAND
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	// set seed
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerateUniform(gen, dRDisp, imgSize);
	
	cudaDeviceSynchronize();

	if(t>0)
		random_end[t-1] = GetTimer();
	
	cudaDeviceSynchronize();

	if(t>0)
	{
		StartTimer();
		main_start[t-1] = 0.0;
	}
	
	for(int i=0; i<iteration; i++)
	{	
		stereoMatching<<<gridSize, blockSize>>>(dRDisp, dRPlanes, cols, rows, states, i);	 

		cudaDeviceSynchronize();
	}

	if(t>0)
		main_end[t-1] = GetTimer();

	}


	// copy disparity map from global memory on device to host
	cudaMemcpy(lImgPtr_f, dRDisp, sizeof(float)*cols*rows, cudaMemcpyDeviceToHost);

	//float to char
	imgCharToFloat(lImgPtr_8u, lImgPtr_f, true);
	
	double average = 0.0, sd = 0.0;
	timingStat(random_start, random_end, nt, &average, &sd);

	cout<<"initial random disp: "<<average<<"ms  sd"<<sd<<endl;

	timingStat(main_start, main_end, nt, &average, &sd);

	cout<<"main: "<<average<<"ms  sd"<<sd<<endl;


	// Free device memory
	cudaFree(dLImgPtr_f);	
	cudaFree(dRImgPtr_f);
	cudaFree(dRDisp);
	cudaFree(dRPlanes);
	cudaFree(states);
	curandDestroyGenerator(gen);
	cudaDeviceReset();

	// write image
	writePGM("disp_cuda_10iter.pgm", lImgPtr_8u);

	delete[] lImgPtr_8u;
	delete[] rImgPtr_8u;
	delete[] lImgPtr_f;
	delete[] rImgPtr_f;
	
	return 0;
}

