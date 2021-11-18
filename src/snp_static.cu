#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <snp_static.hpp>
#include "error_check.hpp"

using namespace std;

/** Allocation */
SNP_static_sparse::SNP_static_sparse(uint n, uint m) : SNP_model(n,m)
{
    // n is num of rows, m is num of colums. 
    // done by subclasses
    this->trans_matrix    = (int*)  malloc(sizeof(int)*n*m);
    this->spiking_vector  = (int*) malloc(sizeof(int)*m); // spiking vector

    memset(this->trans_matrix,0,sizeof(int)*n*m);

    cuda_check(cudaMalloc(&this->d_trans_matrix,  sizeof(int)*n*m));
    cuda_check(cudaMalloc(&this->d_spiking_vector,sizeof(int)*m));

    cuda_check(cudaMemset(this->d_spiking_vector, -1, sizeof(int)*m));
}

SNP_static_ell::SNP_static_ell(uint n, uint m) : SNP_model(n,m)
{
    //Allocate cpu variables
    this->trans_matrix    = (int*)  malloc(sizeof(int)*n*m*2);
    this -> spiking_vector = (int*) malloc(sizeof(int)*m);
    this->z_vector    = (int*) malloc(sizeof(int)*m);
    
    memset(this->trans_matrix,-1,sizeof(int)*n*m*2);
    memset(this->z_vector,0,sizeof(int)*m);
    this->z = 0;

    //Allocate device variables
    cuda_check(cudaMalloc((&this->d_spiking_vector),  sizeof(int)*m));

    cuda_check(cudaMemset(this->d_spiking_vector, -1, sizeof(int)*m));
    //trans_matrix allocated when z is known
}

SNP_static_optimized::SNP_static_optimized(uint n, uint m) : SNP_model(n,m)
{
    //Allocate cpu variables
    this->trans_matrix    = (int*)  malloc(sizeof(int)*n*n);
    this -> spiking_vector = (int*) malloc(sizeof(int)*n);
    this->z_vector    = (int*) malloc(sizeof(int)*n);
    
    memset(this->trans_matrix,-1,sizeof(int)*n*n);
    memset(this->z_vector,0,sizeof(int)*n);
    this->z=0;

    //Allocate device variables
    cuda_check(cudaMalloc((&this->d_spiking_vector),  sizeof(int)*n));
    //d_trans_matrix allocated when z is known

    cuda_check(cudaMemset(this->d_spiking_vector, -1, sizeof(int)*n));
}

SNP_static_cublas::SNP_static_cublas(uint n, uint m) : SNP_model(n,m,true)
{
    //Allocate cpu variables
    cuda_check(cublasCreate(&this->handle));
    this->neuron_to_include = 0;
    this->trans_matrix    = (float*)  malloc(sizeof(float)*n*m);
    this -> spiking_vector = (float*) malloc(sizeof(float)*m);

    memset(this->trans_matrix,0,sizeof(float)*n*m);

    //Allocate device variables
    cuda_check(cudaMalloc(&this->d_spiking_vector,  sizeof(float)*m));
    cuda_check(cudaMalloc(&this->d_trans_matrix,  sizeof(float)*n*m));
}

SNP_static_cusparse::SNP_static_cusparse(uint n, uint m) : SNP_model(n,m,true)
{
    this->alpha = 1.0f;
    this->beta = 1.0f;
    this->nnz=0;
    //Allocate cpu variables
    cuda_check(cusparseCreate(&this->handle));
    this->neuron_to_include = 0;
    this->trans_matrix    = (int*)  malloc(sizeof(int)*n*m);
    this -> spiking_vector = (float*) malloc(sizeof(float)*m);

    memset(this->trans_matrix,0,sizeof(int)*n*m);

    //Allocate device variables
    cuda_check(cudaMalloc(&this->d_spiking_vector,  sizeof(float)*m));
    cuda_check(cudaMalloc(&this->d_trans_matrix, sizeof(int)*n*m));

    cuda_check( cusparseCreateDnVec(&(this->cse_confv), n, this->df_conf_vector, CUDA_R_32F) );
    cuda_check( cusparseCreateDnVec(&(this->cse_spkv), m, this->d_spiking_vector, CUDA_R_32F) );
}

/** Free mem */
SNP_static_ell::~SNP_static_ell()
{
    free(this->z_vector);
}

SNP_static_optimized::~SNP_static_optimized()
{
    free(this->z_vector);
}

SNP_static_cublas::~SNP_static_cublas()
{
    free(this->f_conf_vector);
    cudaFree(this->df_conf_vector);
    cudaFree(this->d_spiking_vector);
    cudaFree(this->d_trans_matrix);
    cublasDestroy(this->handle);
}

SNP_static_cusparse::~SNP_static_cusparse()
{
    free(this->f_conf_vector);
    cudaFree(this->df_conf_vector);
    cudaFree(this->d_spiking_vector);
    cudaFree(this->d_trans_matrix);
    cuda_check(cusparseDestroySpMat(cse_trans_mx));
    cuda_check(cusparseDestroyDnVec(cse_confv));
    cuda_check(cusparseDestroyDnVec(cse_spkv));
    cuda_check(cusparseDestroy(this->handle));
    cuda_check( cudaFree(&this->d_csrOffsets));
    cuda_check( cudaFree(&this->d_csrColumns));
    cuda_check( cudaFree(&this->d_csrValues));
}

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

void SNP_static_ell::print_transition_matrix(){
    assert(z > 0);
    printf("Transition matrix\n");

    for(int i=0; i<z; i++){
        for(int j=0; j<m; j++){
            int idx = (i*m*2 + j*2);
            printf("(%d, %d)",trans_matrix[idx], trans_matrix[idx+1]);
        }  
        printf("\n");
    }
    printf("\n");
}

void SNP_static_optimized::print_transition_matrix(){
    assert(z > 0);
    printf("Transition matrix\n");

    for(int i=0; i<z; i++){
        for(int j=0; j<n; j++){
            int idx = (i*n + j);
            printf("%d ",trans_matrix[idx]);
        }  
        printf("\n");
    }
    printf("\n");
}

void SNP_static_cublas::print_transition_matrix(){
    printf("Transition matrix\n");
    for (int j=0; j<m; j++){
		for (int i=0; i<n; i++){
            printf("%.1f ",trans_matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void SNP_static_cusparse::print_transition_matrix(){
    printf("Transition matrix\n");
    for (int j=0; j<m; j++){
		for (int i=0; i<n; i++){
            printf("%d ",trans_matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void k_print_trans_mx_sparse(int *mx, int n, int m){
    printf("Transition matrix(gpu memory)\n");

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            printf("%d ",mx[i*n + j]);
        }  
        printf("\n");
    }
    printf("\n");
}


__global__ void k_print_spk_v_generic(int *spkv, int m){
    printf("Spiking vector\n");
    for(int i=0; i<m; i++){
        printf("%d ", spkv[i]);
    }
    printf("\n");
}

__global__ void k_print_spk_v_generic(float *spkv, int m){
    printf("Spiking vector\n");
    for(int i=0; i<m; i++){
        printf("%.1f ", spkv[i]);
    }
    printf("\n");
}

__global__ void k_print_dys_v_generic(uint *dys, int n){
    printf("Delays vector\n");
    for(int i=0; i<n; i++){
        printf("%d ", dys[i]);
    }
    printf("\n");
}

void SNP_static_sparse::print_spiking_vector(){
    //print from gpu
    // k_print_trans_mx_sparse<<<1,1>>>(d_trans_matrix, n, m);
    k_print_spk_v_generic<<<1,1>>>(d_spiking_vector, m);
    cudaDeviceSynchronize();
}

void SNP_static_ell::print_spiking_vector(){
    k_print_spk_v_generic<<<1,1>>>(d_spiking_vector, m);
    cudaDeviceSynchronize();
}

void SNP_static_optimized::print_spiking_vector(){
    k_print_spk_v_generic<<<1,1>>>(d_spiking_vector, n);
    cudaDeviceSynchronize();
}

void SNP_static_cublas::print_spiking_vector(){
    k_print_spk_v_generic<<<1,1>>>(d_spiking_vector, m);
    cudaDeviceSynchronize();
}

void SNP_static_cusparse::print_spiking_vector(){
    k_print_spk_v_generic<<<1,1>>>(d_spiking_vector, m);
    cudaDeviceSynchronize();
}

void SNP_static_sparse::print_delays_vector(){
    k_print_dys_v_generic<<<1,1>>>(d_delays_vector, n);
    cudaDeviceSynchronize();
}

void SNP_static_ell::print_delays_vector(){
    k_print_dys_v_generic<<<1,1>>>(d_delays_vector, n);
    cudaDeviceSynchronize();
}

void SNP_static_optimized::print_delays_vector(){
    k_print_dys_v_generic<<<1,1>>>(d_delays_vector, n);
    cudaDeviceSynchronize();
}

void SNP_static_cublas::print_delays_vector(){
    //intentionally left blank
}

void SNP_static_cusparse::print_delays_vector(){
    //intentionally left blank
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
                conf_vector[nid]-=rc[r];
                
                /*handle situation where rule was previously selected, neuron had to wait d steps, but right when it is 
                about to be fired same rule is selected, losing the action of the first time it was selected.
                When this situation appears in rule r, spiking vector[r] is incremented by one, meaning one instance of rule 
                r has already completed delay time and is waiting to be fired. For performance purposes, all instances 
                of rules in this situation will be applied together when its delay counter gets to 0 */

                if(spiking_vector[r] > 0 && delays_vector[nid] == 0){
                    spiking_vector[r]+=1;
                }else{
                    spiking_vector[r] = 1;
                }
                
                delays_vector[nid] = rd[r];
                break;
            }
        }
    }
}

__global__ void kalc_spiking_vector_for_optimized(int* spiking_vector, uint* conf_vector, int* rule_index, uint* rei, uint* ren, uint* rc, uint* rd, uint* delays_vector, uint n)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid<n && delays_vector[nid]==0) {
        //vector<int> active_rule_idxs_ni;
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar i = rei[r];
            uchar n = ren[r];
            int x = conf_vector[nid];
            if (((int) (i&(x==n)) || ((1-i)&(x>=n)))){
                //active_ridx.push_back(r);
                delays_vector[nid] = rd[r];
                conf_vector[nid]-=rc[r];
                spiking_vector[nid] = r;
                break;
            }
        }
    }
}

__global__ void kalc_spiking_vector_for_libs(float* spiking_vector, float *conf_vector, int* rule_index, uint* rei, uint* ren, uint n)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid<n) {
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar e_i = rei[r];
            uchar e_n = ren[r];
            int x = conf_vector[nid];
            if ((int) (e_i&(x==e_n)) || ((1-e_i)&(x>=e_n))) {
                spiking_vector[r] = 1;
                break;
            }
        }
    }
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

void SNP_static_ell::calc_spiking_vector() 
{
    uint bs = 256;
    uint gs = (m+255)/256;
    
    kalc_spiking_vector_generic<<<gs,bs>>>(d_spiking_vector, d_conf_vector, d_rule_index, d_rules.Ei, d_rules.En, d_rules.c, d_rules.d, d_delays_vector, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

void SNP_static_optimized::calc_spiking_vector() 
{
    uint bs = 256;
    uint gs = (m+255)/256;
    kalc_spiking_vector_for_optimized<<<gs,bs>>>(d_spiking_vector, d_conf_vector, d_rule_index, d_rules.Ei, d_rules.En, d_rules.c, d_rules.d, d_delays_vector, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

void SNP_static_cublas::calc_spiking_vector() 
{
    //////////////////////////////////////////////////////
    cpu_updated = false;
    //////////////////////////////////////////////////////

    uint bs = 256;
    uint gs = (m+255)/256;
    
    cuda_check(cudaMemset(this->d_spiking_vector, 0, sizeof(float)*m));
    kalc_spiking_vector_for_libs<<<gs,bs>>>(d_spiking_vector, df_conf_vector, d_rule_index, d_rules.Ei, d_rules.En, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

void SNP_static_cusparse::calc_spiking_vector() 
{
    //////////////////////////////////////////////////////
    cpu_updated = false;
    //////////////////////////////////////////////////////

    uint bs = 256;
    uint gs = (m+255)/256;
    
    cuda_check(cudaMemset(this->d_spiking_vector, 0, sizeof(float)*m));
    kalc_spiking_vector_for_libs<<<gs,bs>>>(d_spiking_vector, df_conf_vector, d_rule_index, d_rules.Ei, d_rules.En, n);
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

void SNP_static_ell::include_synapse(uint i, uint j)
{
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        //forgeting rules are not stored in trans_mx. 
        if(rules.p[r]>0){
            trans_matrix[z_vector[r]*m*2+r*2] = j;
            trans_matrix[(z_vector[r]*m*2+r*2)+1] = rules.p[r];
            z_vector[r]++;
        }
    }
}

void SNP_static_optimized::include_synapse(uint i, uint j)
{
    trans_matrix[z_vector[i]*n+i] = j;
    z_vector[i]++;
}

void SNP_static_cublas::include_synapse(uint i, uint j)
{
    //store by columns for better VxM performance
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[i*m+r] = -(float)rules.c[r];  
        trans_matrix[j*m+r] = (float) rules.p[r];
    }
}

void SNP_static_cusparse::include_synapse(uint i, uint j)
{
    //store by columns for better VxM performance
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[i*m+r] = -(int)rules.c[r];  
        trans_matrix[j*m+r] = (int) rules.p[r];
    }
}


void SNP_static_sparse::load_transition_matrix () 
{
    cuda_check(cudaMemcpy(d_trans_matrix, trans_matrix, sizeof(int)*n*m, cudaMemcpyHostToDevice));
}

void SNP_static_ell::load_transition_matrix () 
{
    for(int r=0; r<m; r++){
        int aux_z=z_vector[r];
        if(aux_z>z){
            z=aux_z;
        }
    }
    assert(z>0);

    cuda_check(cudaMalloc((&this->d_trans_matrix),  sizeof(int)*z*m*2));
    cuda_check(cudaMemcpy(d_trans_matrix, trans_matrix, sizeof(int)*z*m*2, cudaMemcpyHostToDevice));
}

void SNP_static_optimized::load_transition_matrix (){

    for(int i=0; i<n; i++){
        int z_aux = z_vector[i];
        if(z_aux>z){
            z = z_aux;    
        }
    }

    // this-> trans_matrix = (int *) realloc(this->trans_matrix,sizeof(int)*n*z);
    cuda_check(cudaMalloc((&this->d_trans_matrix),  sizeof(int)*n*z));
    cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(int)*n*z,  cudaMemcpyHostToDevice);
}

void SNP_static_cublas::load_transition_matrix () 
{
    //if a neuron doesnt include synapses, then rules of that neuron are not included in transition matrix (so forgetting rules are ignored). 
    while(neuron_to_include < n){
        for(int r=rule_index[neuron_to_include]; r<rule_index[neuron_to_include+1]; r++){
            trans_matrix[neuron_to_include*m + r] = -(float)rules.c[r];
        }
        neuron_to_include++;
    }
    cuda_check(cudaMemcpy(d_trans_matrix, trans_matrix, sizeof(float)*n*m, cudaMemcpyHostToDevice));
}

int SNP_static_cusparse::get_nnz (){
    int nnz = 0;
    for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
            if(trans_matrix[i*m + j] != 0){
              nnz++;  
            }
		}
	}

    return nnz;
}

__global__ void gen_CSR_vectors(int * trans_matrix, int nrows, int ncols, int * csrOffsets, int * csrColumns, float* csrValues){
    int i_nz = 0; 
    csrOffsets[0]=0;
    for(int i=0; i<nrows; i++){
        for(int j=0; j<ncols; j++){
            if(trans_matrix[i*ncols + j]!=0){
                csrColumns[i_nz] = j;
                // printf("csrColumn[%d]=%d\n",i_nz,j);
                csrValues[i_nz] = trans_matrix[i*ncols + j];
                // printf("csrValues[%d]=%d\n",i_nz,trans_matrix[i*ncols + j]);
                i_nz++;
            }
        }
        csrOffsets[i+1]=i_nz;
        // printf("csrOffset[%d]=%d\n",i+1,i_nz);
    }
}

void SNP_static_cusparse::load_transition_matrix () 
{
    //if a neuron doesnt include synapses, then rules of that neuron are not included in transition matrix (so forgetting rules are ignored). 
    while(neuron_to_include < n){
        for(int r=rule_index[neuron_to_include]; r<rule_index[neuron_to_include+1]; r++){
            trans_matrix[neuron_to_include*m + r] = -rules.c[r];
        }
        neuron_to_include++;
    }

    int nnz = get_nnz();
    cuda_check( cudaMalloc(&this->d_csrOffsets,   sizeof(int)*n));
    cuda_check( cudaMalloc(&this->d_csrColumns,   sizeof(int)*nnz));
    cuda_check( cudaMalloc(&this->d_csrValues,   sizeof(float)*nnz));
    cuda_check(cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(int)*n*m,  cudaMemcpyHostToDevice)); 
    gen_CSR_vectors<<<1,1>>>(d_trans_matrix, n, m, d_csrOffsets,d_csrColumns, d_csrValues);
    cudaDeviceSynchronize();

    cuda_check( cusparseCreateCsr(&this->cse_trans_mx, n, m, nnz,
    d_csrOffsets, d_csrColumns, d_csrValues,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    

    size_t buffer_size;
    cuda_check( cusparseSpMV_bufferSize(
    this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, this->cse_trans_mx, this->cse_spkv, &beta, this->cse_confv, CUDA_R_32F,
    CUSPARSE_MV_ALG_DEFAULT, &buffer_size) );
    cuda_check( cudaMalloc(&this->d_buffer,   buffer_size));
}

__global__ void kalc_transition_sparse(int* spiking_vector, int* trans_matrix, uint* conf_vector,uint * delays_vector, uint * rnid , uint n, uint m){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid<n && delays_vector[nid]==0){
        for (int r=0; r<m; r++){
            
            if(spiking_vector[r] != -1 && delays_vector[rnid[r]]==0){
                conf_vector[nid] += spiking_vector[r] * trans_matrix[r*n+nid];
            }
            
            __syncthreads(); //disable rule when all threads have finished processing row (using only one thread)
            if(nid==0 && spiking_vector[r] != -1 && delays_vector[rnid[r]]==0){
                spiking_vector[r] = -1; 
            }
            
        }
    }
}

__global__ void kalc_transition_ell(int* spiking_vector, int* trans_matrix, uint* conf_vector,uint * delays_vector, uint * rnid , uint z, uint m){
    int rid = threadIdx.x+blockIdx.x*blockDim.x;
    if (rid<m && spiking_vector[rid]>0 && delays_vector[rnid[rid]]==0){
        for(int i=0; i<z; i++){
            int neuron = trans_matrix[m*2*i+rid*2];
            int value = trans_matrix[m*2*i+rid*2+1];
            if(neuron==-1 && value==-1){
                break;
            }
            if(delays_vector[neuron]==0){
                //mult value times number of followed activation of a rule (ie. spiking_vector[rid])
                atomicAdd((uint *)&conf_vector[neuron], (uint)value*spiking_vector[rid]);
            }        
        }
        spiking_vector[rid] = -1;
    }
}

__global__ void kalc_transition_optimized(int* spiking_vector, int* trans_matrix, uint* conf_vector, uint* delays_vector, uint* rc, uint* rp, int z, uint n){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;

    if(nid<n && spiking_vector[nid]>=0 && delays_vector[nid]==0){
        int rid = spiking_vector[nid];
        int p = rp[rid];
        // printf("nid:%d, rid:%d, c:%d, p:%d\n", nid, rid, c, p);

        for(int j=0; j<z; j++){
            int n_j = trans_matrix[j*n+nid]; //nid is connected to n_j. 

            if(n_j >= 0){
                if(delays_vector[n_j]>0) break;
                atomicAdd((int *) &conf_vector[n_j], p);
            }else{
                //if padded value (-1)
                break;
            }
        }
        spiking_vector[nid]= -1;
    }
}

__global__ void update_delays_vector_generic(uint *delays_vector, uint n){
    
    int nid = threadIdx.x+blockIdx.x*blockDim.x;
    if(nid<n && delays_vector[nid]>0){
        delays_vector[nid]--;
    }
}

__global__ void k_check_next_trans(bool *calc_nxt, int* spkv, int spkv_size, uint * delays, int neurons){
    calc_nxt[0] = false;
    
    for(int i=0; i<spkv_size; i++){
        if(spkv[i] !=-1){
            calc_nxt[0] = true;
            break;   
        }
    }

    if(!calc_nxt[0]){   
        for(int i=0; i<neurons; i++){
            if(delays[i] > 0){
                calc_nxt[0] = true;
                break;
            }
        }
    }
}

__global__ void k_check_next_trans_for_libs(bool *calc_nxt, float *spkv, int spkv_size){
    calc_nxt[0] = false;
    for(int i=0; i<spkv_size; i++){
        if(spkv[i] != 0){
            calc_nxt[0] = true;
            break;   
        }
    }
}

bool SNP_static_sparse::check_next_trans(){
    k_check_next_trans<<<1,1>>>(d_calc_next_trans, d_spiking_vector, m, d_delays_vector, n);
    cudaDeviceSynchronize();
    cuda_check(cudaMemcpy(calc_next_trans, d_calc_next_trans, sizeof(bool),cudaMemcpyDeviceToHost));
    // printf("calc_next:%d",calc_next_trans[0]);
    return calc_next_trans[0];
}

bool SNP_static_ell::check_next_trans(){
    k_check_next_trans<<<1,1>>>(d_calc_next_trans, d_spiking_vector, m, d_delays_vector, n);
    cudaDeviceSynchronize();
    cuda_check(cudaMemcpy(calc_next_trans, d_calc_next_trans, sizeof(bool),cudaMemcpyDeviceToHost));
    return calc_next_trans[0];
}

bool SNP_static_optimized::check_next_trans(){
    k_check_next_trans<<<1,1>>>(d_calc_next_trans, d_spiking_vector, n, d_delays_vector, n);
    cudaDeviceSynchronize();
    cuda_check(cudaMemcpy(calc_next_trans, d_calc_next_trans, sizeof(bool),cudaMemcpyDeviceToHost));
    return calc_next_trans[0];
}

bool SNP_static_cublas::check_next_trans(){
    k_check_next_trans_for_libs<<<1,1>>>(d_calc_next_trans, d_spiking_vector, n);
    cudaDeviceSynchronize();
    cuda_check(cudaMemcpy(calc_next_trans, d_calc_next_trans, sizeof(bool),cudaMemcpyDeviceToHost));
    return calc_next_trans[0];
}

bool SNP_static_cusparse::check_next_trans(){
    k_check_next_trans_for_libs<<<1,1>>>(d_calc_next_trans, d_spiking_vector, n);
    cudaDeviceSynchronize();
    cuda_check(cudaMemcpy(calc_next_trans, d_calc_next_trans, sizeof(bool),cudaMemcpyDeviceToHost));
    return calc_next_trans[0];
}

void SNP_static_sparse::calc_transition()
{
    //////////////////////////////////////////////////////
    cpu_updated = false;
    //////////////////////////////////////////////////////

    kalc_transition_sparse<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector, d_delays_vector, d_rules.nid,n,m);
    cuda_check(cudaGetLastError());
    update_delays_vector_generic<<<n+255,256>>>(d_delays_vector, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

void SNP_static_ell::calc_transition()
{
    //////////////////////////////////////////////////////
    cpu_updated = false;
    //////////////////////////////////////////////////////

    kalc_transition_ell<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector, d_delays_vector, d_rules.nid,z,m);
    cuda_check(cudaGetLastError());
    update_delays_vector_generic<<<n+255,256>>>(d_delays_vector, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

void SNP_static_optimized::calc_transition()
{
    //////////////////////////////////////////////////////
    cpu_updated = false;
    //////////////////////////////////////////////////////

    kalc_transition_optimized<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector, d_delays_vector, d_rules.c, d_rules.p, z,n);
    cuda_check(cudaGetLastError());
    update_delays_vector_generic<<<n+255,256>>>(d_delays_vector, n);
    cuda_check(cudaGetLastError());
    cudaDeviceSynchronize();
}

void SNP_static_cublas::calc_transition()
{
    float al =1.0f;
    float bet =1.0f;
    
    cuda_check(cublasSgemv(handle,CUBLAS_OP_T,m,n,&al,d_trans_matrix,m,d_spiking_vector,1,&bet,df_conf_vector,1));
}

void SNP_static_cusparse::calc_transition()
{
    
    cuda_check( cusparseSpMV(this->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &this->alpha, cse_trans_mx, cse_spkv, &this->beta, cse_confv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, d_buffer));
}