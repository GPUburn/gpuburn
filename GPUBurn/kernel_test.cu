/*
 * This file includes all kernels designed to perform a specific test of the GPU
 *
 */
 #include <stdio.h>
 
 /*
  * Return SM and Lane identifier.
  */
__device__ int whereami(){
	unsigned int sm, lane;

	asm volatile("mov.u32 %0, %smid;":"=r"(sm));
	asm volatile("mov.u32 %0, %laneid;":"=r"(lane));
	
	return ((int)(1<<30 | (sm << 16) | lane));
}	

	
__global__ void MADKernel(int 	 * vecA,	// Is there an Error and where 
                    	float * vecB,  		// Reference result
                    	float * vecC,		// Input Number
                     	float * vecD,		// Input Number
                     	const uint nbiter)
{
    uint tid = blockIdx.x*blockDim.x + threadIdx.x;
    int i;
	float r0, r1, r2;
    
    r0 = vecD[tid];
    r1 = vecC[tid];
    r2 = vecD[tid];
    
    for(i=0; i<nbiter; i++)
	    r0 = r0 + r1*r2;
	
	if (r0 != vecB[tid])
		vecA[tid] = whereami();
	else 
		vecA[tid] = 0;

	vecB[tid] = r0;
}



#define SIZE_OF_SHARED_MEM 2048
#define NBPATTERN 4
__constant__ uint pattern[NBPATTERN]={0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x00000000};

__global__ void ShMemKernel(int * vecA,			// Is there an Error and where 
                    		float * vecB,  		// Reference result
                    		float * vecC,		// Input Number
                     		float * vecD,		// Input Number
                     		const uint nbiter)
{
    uint gid   = blockIdx.x*blockDim.x + threadIdx.x;
    uint lid   = threadIdx.x;
    uint lsize = blockDim.x;
    int i,j,k;
    __shared__ int localBuffer[SIZE_OF_SHARED_MEM];

    vecA[gid] = 0;
        
    for(i=0; i<nbiter; i++){
    	for(k=0; k < NBPATTERN; k++){
			// 1: Write in shared memory
			for(j=lid; j<SIZE_OF_SHARED_MEM; j+=lsize)
				localBuffer[j]=pattern[k];
				
			//	barrier(CLK_LOCAL_MEM_FENCE);
	
			// 2: Read the value in shared memory
			for(j=lid; j<SIZE_OF_SHARED_MEM; j+=lsize){
				if(localBuffer[j] != pattern[k])
					vecA[gid] = j;
			}
		}

    }
}


__global__ void REGREGKernel(int * vecA,  	// Is there an Error and where
                        	float * vecB,  	// Reference result
                        	float * vecC,  	// Input Number
                        	float * vecD,  	// Input Number
                        	const uint nbiter)
{
    uint tid = blockIdx.x*blockDim.x + threadIdx.x;
    int i;
    float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;
    float r10, r11, r12, r13, r14, r15, r16, r17, r18, r19;
    float r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;
    float r30, r31;

    
    for(i=0;i<nbiter;i++){
		r0= vecC[tid];
		r1= r0; 
		r2= r1; 
		r3= r2; 
		r4= r3; 
		r5= r4; 
		r6= r5; 
		r7= r6; 
		r8= r7;
		r9= r8;
		r10= r9;
		r11= r10;
		r12= r11;
		r13= r12;
		r14= r13;
		r15= r14;
		if (r15 != vecC[tid])
                vecA[tid] = whereami();
        else
                vecA[tid] = 0;

 		r16= vecD[tid]; 
        r17= r16;
        r18= r17;
        r19= r18;
        r20= r19;
        r21= r20;
        r22= r21;
        r23= r22;
        r24= r23;
        r25= r24;
 		r26= r25;
        r27= r26;
        r28= r27;
        r29= r28;
		r30= r29;
		r31= r30;
        if (r31 != vecD[tid])
                vecA[tid] = whereami();
        else
                vecA[tid] = 0;
	}
}

