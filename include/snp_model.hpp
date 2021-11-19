#ifndef _SNP_MODEL_
#define _SNP_MODEL_

// Modes
#define NO_DEBUG 0
#define DEBUG    1

typedef unsigned short int  ushort;
typedef unsigned int        uint;
typedef unsigned char       uchar;

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusparse.h> 

using namespace std;

class SNP_model
{
public:
    SNP_model(uint n, uint m, bool using_lib=false);
    ~SNP_model();
    
    /** indicate verbosity level, number of times the whole
    *   computation is repeated and whether write of the 
    *   output to a file is intended
    */
    void set_snpconfig(int verbosity_lv, int repetitions, char* outfile, bool count_time, bool get_mem_info=false); 
    /** Writes computed configuration to file 
    */
    void write_to_file();
    /** 
     * Set a number of spikes, given by s, in the neuron nid.
     * This method should be used to create the initial configuration.
     * This replaces previous value in that neuron in the configuration */
    void set_spikes (uint nid, uint s);
    /** 
     * Consult number of spikes in neuron nid. */
    uint get_spikes (uint nid);
    /** 
     * Add a rule to neuron nid, 
     * regular expression defined by e_n and e_i, and a^c -> a^p.
     * This must be called sorted by neuron, and before adding synapses */
    void add_rule (uint nid, uint e_n, uint e_i, uint c, uint p, uint d);
    /** 
     * Add synapse from neuron i to j. 
     * This must be called after adding all rules */
    void add_synapse (uint i, uint j);
    /** 
     * Perform a transition step on the model. 
     * Returns if no more steps can be done. */
    bool transition_step(int i=-1);
    /**
     * Prints configuration vector. If cpu not updated, downloads it from gpu first.
     */
    void print_conf_vector(ofstream *fs=NULL);
    /** 
     * Simulate a computation of the model. 
     * Optionally, set a limit to l steps */
    //  void compute(int l=3) { while(l--) {transition_step();} };
    void compute(int i=-1); //(optional) provide a number of times for the exact times the computations will be performed. Pass -1 to ignore parameter.

protected:
    uint n;        // number of neurons
    uint m;        // number of rules

    // CPU part
    uint *delays_vector;
    uint *conf_vector;     // configuration vector (# neurons)
    float *f_conf_vector;
    int *trans_matrix;    // transition matrix (# rules * # neurons)
    int *spiking_vector;  // spiking vector (# neurons)
    int   *rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)

    struct _rule {
        uint  *En;          // indicates for each rule, the regular expression multiplicity
        uint  *Ei;          // indicates for each rule, the regular expression type
        uint  *c;           // indicates for each rule, the LHS
        uint  *p;           // indicates for each rule, the RHS
        uint   *d;
        uint   *nid;         // indicates for each rule, the corresponding neuron (#rules)
    } rules, d_rules;

    // GPU counterpart
    uint *d_delays_vector;
    uint *d_conf_vector;
    float *df_conf_vector;
    int *d_trans_matrix;
    int *d_spiking_vector;
    int *d_rule_index;      // indicates for each neuron, the starting rule index (# neurons+1)

    // Consistency flags
    bool gpu_updated;           // true if GPU copy is updated
    bool cpu_updated;           // true if CPU copy is updated
    bool done_rules;            // true if all rules have been introduced (preventing adding synapses)
    bool *calc_next_trans;      // true if next transition has to be calculated (stored in slot 0). Host variable
    bool *d_calc_next_trans;    // Device variable
    bool using_lib;             // true if using either cublas or cusparse libraries
    
    // Config variables
    int verbosity_lv;
    int repetitions;
    char *outfile;
    bool count_time;
    int step;                   // Step in which the computation is in
    bool get_mem_info;

    // auxiliary methods
    /** 
     * Load the introduced model to the GPU.
     * The status of model computation gets reset */
    void load_to_gpu();
    /** 
     * Download information from the GPU. */
    void load_to_cpu();  
    /**
     * Calculates the spiking vector with the current configuration */

    // auxiliary virtual methods (to be defined in the different simulators)    
    // @override define this method to include a synapse in the transition matrix 
    virtual void include_synapse(uint i, uint j) = 0;
    // @override define this method to send the transition matrix to GPU
    virtual void load_transition_matrix() = 0;
    // @override define method to obtain spiking vector
    virtual void calc_spiking_vector() = 0;
    // @override define method to check if next transitions needs to be calculated. */
    virtual bool check_next_trans() = 0;
    // @override define this method to compute the transition, once the spiking vector is calculated
    virtual void calc_transition() = 0;
    // @override define this method to print the transition matrix
    virtual void print_transition_matrix() = 0;
    // @override define this method to print the spiking vector
    virtual void print_spiking_vector() = 0;
    // @override define this method to print the delays vector
    virtual void print_delays_vector() = 0;
};


#endif
