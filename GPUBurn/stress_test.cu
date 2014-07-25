#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>  
#include <vector> 
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <unistd.h>    /* for getopt */

#include "stress_test.h"
#include "kernel_test.cu"

//#include <unistd.h>		// for sleep()

StressTest::~StressTest(){
  if (output_file!=&std::cout) 
     delete output_file;
}

StressTest::StressTest(){
	// Default parameters.
	devID = 0;
	kernelID = -1;
	allkernel=false;
	h_iter = 1;
	TimeLimit = 5;
	maxtemp = 200;
	timewait = 0;
	output_file = &std::cout;
}

void 
StressTest::parse_command_line(int argc, char **argv){
    int c;
    
    while ( (c = getopt(argc, argv, "ad:f:k:t:N:hm:")) != -1) {
        switch (c) {
        case 'a':
        	allkernel = true;
        	break;
        case 'd':
        	devID = atoi(optarg);
        	break;
        case 'f':
        	output_file = new std::ofstream(optarg);
        	std::cout << "Opening file :" << optarg <<"\n";
        	break;
        case 'k':
		    kernelID = -1;
		    if(strcmp(optarg,"ALL") == 0){
		    	kernelID = 0;
		    	allkernel = true;
		    }
	        for (int i=0; i<NB_KERNEL; i++)
    			if(strcmp(KernelNames[i],optarg) == 0) kernelID = i;
			if (kernelID == -1)
				std::cerr << "ERROR: (" << optarg << ") is an invalid Kernel Name" << std::endl;
            break;
        case 't':
            TimeLimit = atoi(optarg);
            break;
        case 'm':
            maxtemp = atoi(optarg);
            break;
        case 'N':
        	h_iter = atoi(optarg);
            break;
        case 'h':
        case '?':
        default:
			std::cerr << "Usage: " << argv[0] << " [-d devID] [-f FILENAME] [-k KERNEL] [-t timelimit] [-N Iter] \n" ;
			std::cerr << " -a : Test all kernel alternatively \n";			
			std::cerr << " -d : Select the device to test \n";			
			std::cerr << " -f : Specify the output file for the result (default stdout) \n";			
			std::cerr << " -m : Specify the maximum temperature of the device (default 200°C, not implemented on this version) \n";			
			std::cerr << " -k : Specify which kernel to select (ALL, ";
			for(int i=0; i<NB_KERNEL; i++)
				std::cerr << KernelNames[i] << ", ";
			std::cerr <<") \n";	
			std::cerr << " -t : Set the timer to 'timelimit' second (default 5 sec, -1 for nolimit)\n";  
			std::cerr << " -N : Set the number of iteration per work-item to 'Iter' (default 1000) \n";
			std::cerr << " -h : Print out this help \n";			
			exit(-1);         	
            break;
        }
    }
    if (optind < argc) {
        std::cerr << "non-option ARGV-elements: ";
        while (optind < argc)
        	std::cerr << argv[optind++] << " ";
        std::cerr << std::endl;
        exit(-1);         	
    }
}

void
StressTest::DeviceInit(){
	cudaSetDevice(devID);
	cudaError_t error;
	cudaDeviceProp deviceProp;

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited){
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
	else
		printf("Tested GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);


	multiProcessorCount = deviceProp.multiProcessorCount;
	cudaCore = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		
	std::cout << SEPARATOR ;
	std::cout << " Execution will be launched with 1 Block for each of the " << multiProcessorCount << " Multiprocessors \n";
	std::cout << " with 1 threads for each of the " << cudaCore << " CUDA Cores \n"; 
	std::cout << " Maximum temperature of the device is " << maxtemp << "°C \n";
	std::cout << SEPARATOR ;
}

void
StressTest::InitTest( ) {
	std::cout << SEPARATOR ;
			
	vLen = multiProcessorCount * cudaCore;
	vSize = vLen * sizeof(int);
	
	// set up the kernel inputs
	h_vecA = new int[vLen];
	h_vecB = new float[vLen];
	h_vecC = new float[vLen];
	h_vecD = new float[vLen];

	cudaError_t error;

    error = cudaMalloc((void **) &d_vecA, vSize);

    if (error != cudaSuccess){
        printf("cudaMalloc d_vecA returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_vecB, vSize);

    if (error != cudaSuccess){
        printf("cudaMalloc d_vecB returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_vecC, vSize);

    if (error != cudaSuccess){
        printf("cudaMalloc d_vecC returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_vecD, vSize);

    if (error != cudaSuccess){
        printf("cudaMalloc d_vecD returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }	
}

inline void 
StressTest::Init_Data_and_Argument(int seed) {
//	std::cout << "Start Data Initialization (vector lenght:" << vLen <<")\n";
	srand(seed);
	for(int i=0; i < vLen; i++) h_vecA[i] = 0; //rand();
	for(int i=0; i < vLen; i++) h_vecB[i] = 0; //rand();
	for(int i=0; i < vLen; i++) h_vecC[i] = 1.+((float)rand() / (float)RAND_MAX);
	for(int i=0; i < vLen; i++) h_vecD[i] = 1.+((float)rand() / (float)RAND_MAX);
	cudaError_t error;
	  
    error = cudaMemcpy(d_vecA, h_vecA, vSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMemcpy (d_vecA,h_vecA) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_vecB, h_vecB, vSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMemcpy (d_vecB,h_vecB) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_vecC, h_vecC, vSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMemcpy (d_vecC,h_vecC) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_vecD, h_vecD, vSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("cudaMemcpy (d_vecD,h_vecD) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }	
}

inline void 
StressTest::CheckError(uint kernelidx) {
	time_t tmptime;

	cudaError_t error;
	
	// bring data back to the host via a blocking read
    error = cudaMemcpy(h_vecA, d_vecA, vSize, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess){
        printf("cudaMemcpy (h_vecA,d_vecA) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	for(int i=0; i < vLen; i++){
		if (h_vecA[i] != 0){
			time(&tmptime);
			*output_file << asctime(localtime(&tmptime)) << ": " << KernelNames[kernelidx] 
						 <<", ERROR after " << difftime(time(NULL),tbegin) 
						 << "sec. at (SM: " << ((h_vecA[i] & 0x0FFF0000) >> 16) 
						 << " ; LaneId: " << (h_vecA[i] & 0x0000FFFF) << ")"
						 <<" \n";	
		}
	}

	(*output_file).flush();
}


inline void 
StressTest::ExecKernel(int kID){
    // Setup execution parameters
    dim3 threads(cudaCore);
    dim3 grid(multiProcessorCount);

	switch(kernelID){
		case 0: // MADKernel
			MADKernel<<<grid,threads>>>(d_vecA, d_vecB, d_vecC, d_vecD, h_iter); 
			break;
		case 1: // ShMemKernel
			ShMemKernel<<<grid,threads>>>(d_vecA, d_vecB, d_vecC, d_vecD, h_iter); 
			break;
		case 2: // REGREGKernel
			REGREGKernel<<<grid,threads>>>(d_vecA, d_vecB, d_vecC, d_vecD, h_iter); 
			break;
	}

	cudaDeviceSynchronize();	
}

void 
StressTest::ExecuteTest(){
	std::cout << "Start testing the device for " << TimeLimit << " seconds \n";	
		
	// Main Loop
	int dd=0;	
	int ker_begin, ker_end;
	
	if (allkernel){
		ker_begin = 0;
		ker_end   = NB_KERNEL;		
	}else{
		ker_begin = kernelID;
		ker_end   = kernelID+1;		
	}
	int count; 		// Variable used to track the number of tests done per generated random set
	for(uint i=ker_begin; i<ker_end; i++){
		// Initialize the start timer
		tbegin=time(NULL);
		std::cout << SEPARATOR ;
		std::cout << "Start Time: " << asctime(localtime(&tbegin)) << std::endl;
		std::cout << "for "<< KernelNames[i] << " : ";	
		
		while((difftime(time(NULL),tbegin) < TimeLimit) || (TimeLimit == -1)){
			if (difftime(time(NULL),tbegin) != dd){
				dd = difftime(time(NULL),tbegin);
				std::cerr << dd << "(" << count<<"), ";
				
				// Test with new set of random values
				Init_Data_and_Argument(time(NULL));
				count = 0;
				// Warm up the device
				ExecKernel(i);	
	
				// check if every results are representable
				// bring data back to the host via a blocking read
				cudaError_t error;
	
				// bring data back to the host via a blocking read
				error = cudaMemcpy(h_vecB, d_vecB, vSize, cudaMemcpyDeviceToHost);

				if (error != cudaSuccess){
					printf("cudaMemcpy (h_vecA,d_vecA) returned error code %d, line(%d)\n", error, __LINE__);
					exit(EXIT_FAILURE);
				}
	
				for(int i=0; i < vLen; i++){
					if (!std::isnormal(h_vecB[i])){
						std::cout << "WARNING: A value at line " << i<< " is not a valid FP value (" << h_vecB[i] <<") \n"; 
					}
				}	
			}

			// Perform the test
			ExecKernel(i);	
			CheckError(i);
			count ++;
		}
		std::cout <<"\n";
	}
}
