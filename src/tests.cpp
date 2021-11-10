#include "tests.hpp"
#include <iostream>
#include <new>

// Algorithms

#define NO_COMPRESSION		0
#define ELL 				1
#define OPTIMIZED			2
// #define GPU_CUBLAS 			3
// #define GPU_CUSPARSE	 		4


SNP_model* init_alg(int algorithm, int neurons, int rules){
	switch (algorithm)
	{
		case NO_COMPRESSION:
			return new SNP_static_sparse(neurons, rules); 
			//init_params(MAP1,n,0.15,DEBUG,algorithm,&params);
			//init_vars(8,10,&params,&vars);
		break;
		case ELL:
			//init_params(MAP2,n,0.15,DEBUG,algorithm,&params);
			//init_vars(32,9.3,&params,&vars);
		break;
		case OPTIMIZED:
			//init_params(MAP3,n,0.15,DEBUG,algorithm,&params);
			//init_vars(21.5,21.5,&params,&vars);
		break;		
		default:
			printf("Invalid algorithm\n");
			exit(0);
	}

	return NULL;
}


void simple_snp(int alg, int verbosity_lv, int repetitions, char* outfile){
	int neurons = 3;
    int rules = 5; 
	SNP_model *snp = init_alg(alg, neurons, rules);
    snp->set_snpconfig(verbosity_lv, repetitions, outfile);
    int C0[3] = {2,1,1};
	
    for (int i=0; i<neurons; i++){
		snp->set_spikes (i, C0[i]);
	}

    //add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
	snp->add_rule(0, 2, 1, 1, 1, 0);
	snp->add_rule(0, 2, 1, 2, 1, 0);
	snp->add_rule(1, 1, 1, 1, 1, 0);
	snp->add_rule(2, 1, 1, 1, 1, 0);
	snp->add_rule(2, 2, 1, 2, 0, 0);

    snp->add_synapse(0,1);
	snp->add_synapse(1,0);
	snp->add_synapse(0,2);
	snp->add_synapse(1,2);

    snp->transition_step(); 

	delete snp;
}

