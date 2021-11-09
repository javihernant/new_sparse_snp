#include <stdio.h>
#include <stdlib.h>
#include <assert.h> //#define assert
#include <cuda.h>
#include "snp_model.hpp"
#include "error_check.hpp"

using namespace std;

/** Allocation */
SNP_model::SNP_model(uint n, uint m)
{
    // allocation in CPU
    this->n = n;  // number of neurons
    this->m = m;  // number of rules
    this->conf_vector     = (uint*) malloc(sizeof(uint)*n); // configuration vector (only one, we simulate just a computation)
    this->delays_vector = (uint*) malloc(sizeof(uint)*n); 
    this->rule_index      = (uint*)   malloc(sizeof(uint)*(n+1)); // indeces of rules inside neuron (start index per neuron)
    this->rules.Ei        = (uint*)  malloc(sizeof(uint)*m); // Regular expression Ei of a rule
    this->rules.En        = (uint*)  malloc(sizeof(uint)*m); // Regular expression En of a rule
    this->rules.c         = (uint*)  malloc(sizeof(uint)*m); // LHS of rule
    this->rules.p         = (uint*)  malloc(sizeof(uint)*m); // RHS of rule
    this->rules.d         = (uint*)  malloc(sizeof(uint)*n); // RHS of rule
    this->rules.nid       = (uint*)   malloc(sizeof(uint)*(m)); // Index of the neuron where the rule is

    // allocation in GPU
    cudaMalloc(&this->d_conf_vector,   sizeof(uint)*n);
    cudaMalloc(&this->d_delays_vector,   sizeof(uint)*n);
    cudaMalloc(&this->d_rule_index,    sizeof(uint)*(n+1));
    cudaMalloc(&this->d_rules.Ei,      sizeof(uint)*m);
    cudaMalloc(&this->d_rules.En,      sizeof(uint)*m);
    cudaMalloc(&this->d_rules.c,       sizeof(uint)*m);
    cudaMalloc(&this->d_rules.p,       sizeof(uint)*m);
    cudaMalloc(&this->d_rules.d,       sizeof(uint)*m);
    cudaMalloc(&this->d_rules.nid,     sizeof(uint)*m);

    // initialization (only in CPU, having updated version)
    memset(this->conf_vector,   0,  sizeof(uint)*n);
    memset(this->d_delays_vector,   0,  sizeof(uint)*n);
    memset(this->rule_index,    0,  sizeof(uint)*(n+1));
    memset(this->rules.Ei,      0,  sizeof(uint)*m);
    memset(this->rules.En,      0,  sizeof(uint)*m);
    memset(this->rules.c,       0,  sizeof(uint)*m);
    memset(this->rules.p,       0,  sizeof(uint)*m);
    memset(this->rules.d,       0,  sizeof(uint)*n);
    memset(this->rules.nid,     0,  sizeof(uint)*(m));
   
    // memory consistency, who has the updated copy?
    gpu_updated = false; cpu_updated = true;
    done_rules = false;
}

/** Free mem */
SNP_model::~SNP_model()
{
    free(this->conf_vector);
    free(this->spiking_vector);
    free(this->trans_matrix);
    free(this->rule_index);
    free(this->rules.Ei);
    free(this->rules.En);
    free(this->rules.c);
    free(this->rules.p);
    free(this->rules.d);
    free(this->rules.nid);

    cudaFree(this->d_conf_vector);
    cudaFree(this->d_spiking_vector);
    cudaFree(this->d_trans_matrix);
    cudaFree(this->d_rule_index);
    cudaFree(this->d_rules.Ei);
    cudaFree(this->d_rules.En);
    cudaFree(this->d_rules.c);
    cudaFree(this->d_rules.p);
    cudaFree(this->d_rules.d);
    cudaFree(this->d_rules.nid);
}

void SNP_model::set_snpconfig (int verbosity_lv, int repetitions, char *outfile){
    this->verbosity_lv = verbosity_lv;
    this->repetitions = repetitions;
    this->outfile = outfile;
}

void SNP_model::set_spikes (uint nid, uint s)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) load_to_cpu();
    gpu_updated = false;
    //////////////////////////////////////////////////////

    conf_vector[nid] = s;    
}

uint SNP_model::get_spikes (uint nid)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) load_to_cpu();
    //////////////////////////////////////////////////////

    return conf_vector[nid];
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void SNP_model::add_rule (uint nid, uint e_n, uint e_i, uint c, uint p, uint d) 
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    assert(!done_rules);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    if (rule_index[nid+1] == 0) // first rule in neuron
        rule_index[nid+1] = rule_index[nid] + 1; 
    else   // keep accumulation
        rule_index[nid+1] = rule_index[nid+1] - rule_index[nid] + 1; 

    uint rid = rule_index[nid+1]-1;

    rules.Ei[rid] = e_i;
    rules.En[rid] = e_n;
    rules.c[rid]  = c;
    rules.p[rid]  = p;
    rules.nid[rid]= nid;
}

/** Add synapse from neuron i to j. 
    Must be called after adding all rules */
void SNP_model::add_synapse (uint i, uint j) 
{
    //////////////////////////////////////////////////////
    // ensure parameters within limits
    assert(i < n && j < n);
    // ensure all rules have been introduced already
    assert(rule_index[n+1]==m);
    // SNP does not allow self-synapses
    assert(i!=j);
    done_rules = true; // from now on, no more rules can be added
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    include_synapse(i,j);
}

bool SNP_model::transition_step ()
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (!gpu_updated) load_to_gpu();
    cpu_updated = false;
    //////////////////////////////////////////////////////

    calc_spiking_vector();
    calc_transition();
    if(this->verbosity_lv == 1){
        load_to_cpu();
        for(int i=0; i<n; i++){
            printf("%d",conf_vector[i]);
        }
        printf("\n");
    }

    return true; // TODO: check if a stopping criterion has been reached
}

void SNP_model::load_to_gpu () 
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated) return;
    gpu_updated = true;
    //////////////////////////////////////////////////////

    cudaMemcpy(d_conf_vector,   conf_vector,    sizeof(uint)*n,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_rule_index,    rule_index,     sizeof(uint)*(n+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.Ei,      rules.Ei,       sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.En,      rules.En,       sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.c,       rules.c,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.p,       rules.p,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.d,       rules.d,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules.nid,     rules.nid,      sizeof(uint)*m,     cudaMemcpyHostToDevice);

    load_spiking_vector();
    load_transition_matrix();
}

void SNP_model::load_to_cpu ()
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (cpu_updated) return;
    cpu_updated = true;
    //////////////////////////////////////////////////////

    cudaMemcpy(conf_vector, d_conf_vector, sizeof(uint)*n, cudaMemcpyDeviceToHost);
}



