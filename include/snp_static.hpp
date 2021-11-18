#ifndef _SNP_MODEL_STATIC_
#define _SNP_MODEL_STATIC_

#include "snp_model.hpp"

class SNP_static_sparse: public SNP_model
{
public:
    SNP_static_sparse(uint n, uint m);
    ~SNP_static_sparse();

protected:
    void include_synapse(uint i, uint j);
    void load_transition_matrix();
    void calc_spiking_vector();
    bool check_next_trans();
    void calc_transition();
    void print_transition_matrix();
    void print_spiking_vector();
    void print_delays_vector();

};

class SNP_static_ell: public SNP_model
{
public:
    SNP_static_ell(uint n, uint m);
    ~SNP_static_ell();

protected:
    int *z_vector;
    int z;

    void include_synapse(uint i, uint j);
    void load_transition_matrix();
    void calc_spiking_vector();
    bool check_next_trans();
    void calc_transition();
    void print_transition_matrix();
    void print_spiking_vector();
    void print_delays_vector();

};

class SNP_static_optimized: public SNP_model
{
public:
    SNP_static_optimized(uint n, uint m);
    ~SNP_static_optimized();

protected:
    int *z_vector;
    int z;

    void include_synapse(uint i, uint j);
    void load_transition_matrix();
    void calc_spiking_vector();
    bool check_next_trans();
    void calc_transition();
    void print_transition_matrix();
    void print_spiking_vector();
    void print_delays_vector();

};

class SNP_static_cublas: public SNP_model
{
public:
    SNP_static_cublas(uint n, uint m);
    ~SNP_static_cublas();

protected:

    // CPU part
    float *trans_matrix;    // transition matrix (# rules * # neurons)
    float *spiking_vector;  // spiking vector (# neurons)
    int neuron_to_include;  // neuron whose rules have yet to be included. if neuron_to_include == number of neurons, all neuron's rules have been included.

    // GPU counterpart    
    float *d_trans_matrix;    
    float *d_spiking_vector; 
    cublasHandle_t handle;

    void include_synapse(uint i, uint j);
    void load_transition_matrix();
    void calc_spiking_vector();
    bool check_next_trans();
    void calc_transition();
    void print_transition_matrix();
    void print_spiking_vector();
    void print_delays_vector();

};

class SNP_static_cusparse: public SNP_model
{
public:
    SNP_static_cusparse(uint n, uint m);
    ~SNP_static_cusparse();

protected:

    // CPU part
    float *spiking_vector;  // spiking vector (# neurons)
    int neuron_to_include;  // neuron whose rules have yet to be included. if neuron_to_include == number of neurons, all neuron's rules have been included.
    int nnz;                // non-zero elements of transition matrix
    float alpha;
    float beta;
    // GPU counterpart      
    float *d_spiking_vector; 
    cusparseHandle_t     handle;
    cusparseSpMatDescr_t cse_trans_mx;
    cusparseDnVecDescr_t cse_spkv, cse_confv;
    void *d_buffer;
    int * d_csrOffsets;
    int * d_csrColumns;
    float * d_csrValues;

    void include_synapse(uint i, uint j);
    int get_nnz();
    void load_transition_matrix();
    void calc_spiking_vector();
    bool check_next_trans();
    void calc_transition();
    void print_transition_matrix();
    void print_spiking_vector();
    void print_delays_vector();

};



#endif
