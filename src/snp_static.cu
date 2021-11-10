#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <snp_static.hpp>
#include "error_check.hpp"

using namespace std;

/** Allocation */
SNP_static_sparse::SNP_static_sparse(uint n, uint m) : SNP_model(n,m)
{
    // n is num of rows, m is num of colums. 
    // done by subclasses
    this->trans_matrix    = (uint*)  malloc(sizeof(uint)*n*m);
    this->spiking_vector  = (int*) malloc(sizeof(int)*m); // spiking vector

    memset(this->trans_matrix,0,sizeof(uint)*n*m);
    memset(this->spiking_vector,-1,  sizeof(int)*m);


    cuda_check(cudaMalloc(&this->d_trans_matrix,  sizeof(uint)*n*m));
    cuda_check(cudaMalloc(&this->d_spiking_vector,sizeof(uint)*m));

}

/** Free mem */
// SNP_static_sparse::~SNP_static_cublas()
// {
    
// }

void SNP_static_sparse::print_transition_matrix(){
    printf("Transition matrix\n");

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            printf("%d ",trans_matrix[i*n + j]);
        }  
        printf("\n");
    }
    printf("\n");
}

__global__ void k_print_trans_mx_sparse(uint *mx, int n, int m){
    printf("Transition matrix(gpu memory)\n");

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            printf("%d ",mx[i*n + j]);
        }  
        printf("\n");
    }
    printf("\n");
}


__global__ void k_print_spk_v_general(int *spkv, int m){
    printf("Spiking vector\n");
    for(int i=0; i<m; i++){
        printf("%d ", spkv[i]);
    }
    printf("\n");
}

__global__ void k_print_dys_v_general(uint *dys, int n){
    printf("Delays vector\n");
    for(int i=0; i<n; i++){
        printf("%d ", dys[i]);
    }
    printf("\n");
}

void SNP_static_sparse::print_spiking_vector(){
    //print from gpu
    // k_print_trans_mx_sparse<<<1,1>>>(d_trans_matrix, n, m);
    k_print_spk_v_general<<<1,1>>>(d_spiking_vector, m);
    cudaDeviceSynchronize();
}

void SNP_static_sparse::print_delays_vector(){
    k_print_dys_v_general<<<1,1>>>(d_delays_vector, n);
    cudaDeviceSynchronize();
}

__global__ void kalc_spiking_vector_generic(int* spiking_vector, uint* conf_vector, int* rule_index, uint* rei, uint* ren, uint* rc, uint* rd, uint* delays_vector, uint n)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;

    if (nid<n && delays_vector[nid]==0) {
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar e_i = rei[r];
            uchar e_n = ren[r];
            int x = conf_vector[nid];
            if ((int) (e_i&(x==e_n)) || ((1-e_i)&(x>=e_n))) {   
                spiking_vector[r] = 1;
                conf_vector[nid]-=rc[r];
                delays_vector[nid] = rd[r];
                break;
            }
        }
    }
}

__global__ void kalc_spiking_vector_for_optimized(int* spiking_vector, uint* conf_vector, uint* rei, uint* ren, uint* rc, uint m)
{
    uint r = threadIdx.x+blockIdx.x*blockDim.x;
    //TODO
}

void SNP_static_sparse::calc_spiking_vector() 
{
    //////////////////////////////////////////////////////
    cpu_updated = false;
    //////////////////////////////////////////////////////
    uint bs = 256;
    uint gs = (m+255)/256;
    
    kalc_spiking_vector_generic<<<gs,bs>>>(d_spiking_vector, d_conf_vector, d_rule_index, d_rules.Ei, d_rules.En, d_rules.c, d_rules.d, d_delays_vector, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

void SNP_static_sparse::include_synapse(uint i, uint j)
{
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[r*n+i] = 0;  
        trans_matrix[r*n+j] = rules.p[r];
    }
}

void SNP_static_sparse::load_spiking_vector(){
    cuda_check(cudaMemcpy(d_spiking_vector, spiking_vector, sizeof(int)*m,   cudaMemcpyHostToDevice));
}


void SNP_static_sparse::load_transition_matrix () 
{
    cuda_check(cudaMemcpy(d_trans_matrix, trans_matrix, sizeof(uint)*n*m, cudaMemcpyHostToDevice));

}


/*__global__ void ksmvv (short* a, short* v, short* w, uint m) i
{
    uint n = blockIdx.x;
    uint acum = =0;
    for (uint i=tid; i<m; i+=blockDim.x) {
        acum+=a[i]*v[i];
    }
    __syncthreads();

    // reduce

    if (threadIdx.x==0)
        w[n] = acum;
}*/

__global__ void kalc_transition_sparse(int* spiking_vector, uint* trans_matrix, uint* conf_vector,uint * delays_vector, uint * rnid , uint n, uint m){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid<n && delays_vector[nid]==0){
        for (int r=0; r<m; r++){
            if(spiking_vector[r] != -1){
                conf_vector[nid] += trans_matrix[r*n+nid];
            }
            __syncthreads();
            spiking_vector[r] = 0; //disable rule when all threads have finished processing row
        }
    }
}

__global__ void update_delays_vector_static(uint * delays_vector, uint n){
    
    int nid = threadIdx.x+blockIdx.x*blockDim.x;
    if(nid<n && delays_vector[nid]>0){
        delays_vector[nid]--;
    }
}

void SNP_static_sparse::calc_transition()
{
    //////////////////////////////////////////////////////
    cpu_updated = false;
    //////////////////////////////////////////////////////
    
    // if(verbosity>=3){
    //     printVectors_sp_K<<<1,1,0,this->stream2>>>(d_spiking_vector, m, d_delays_vector, n);
    // }

    kalc_transition_sparse<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector, d_delays_vector, d_rules.nid,n,m);
    cuda_check(cudaGetLastError());
    update_delays_vector_static<<<n+255,256>>>(d_delays_vector, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

