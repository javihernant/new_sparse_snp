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



#endif
