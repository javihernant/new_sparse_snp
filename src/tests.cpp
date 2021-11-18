#include "tests.hpp"
#include <iostream>
#include <new>


SNP_model* init_alg(int algorithm, int neurons, int rules){
	switch (algorithm)
	{
		case NO_COMPRESSION:
			return new SNP_static_sparse(neurons, rules);
		break;
		case ELL:
			return new SNP_static_ell(neurons, rules);
		break;
		case OPTIMIZED:
			return new SNP_static_optimized(neurons, rules);
		break;
		case GPU_CUBLAS:
			return new SNP_static_cublas(neurons, rules);
		break;	
		case GPU_CUSPARSE:
			return new SNP_static_cusparse(neurons, rules);
		break;			
		default:
			printf("Invalid algorithm\n");
			exit(0);
	}

	return NULL;
}


void simple_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info){
	int neurons = 3;
    int rules = 5; 
	SNP_model *snp = init_alg(alg, neurons, rules);
    snp->set_snpconfig(verbosity_lv, repetitions, outfile, count_time, mem_info);
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

    snp->compute(1); 

	delete snp;
}

void sort_numbers_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info){
	int size = 50; //adjust to choose how many numbers are sorted
	int nums[size]; //natural numbers sorted in descended order
	for (int i=size; i>0; i--){
		nums[size-i]=i;
	}
	
	int n= size*3; //number of neurons is number of numbers * 3 layers. 
	int m = size + size*size; //each neuron in the first layer has one rule. Each neuron in the second layer has size (of the array of nums to be sorted) rules. There are "size" neurons in each layer (input, second, output).

	SNP_model *snp = init_alg(alg, n, m);
    snp->set_snpconfig(verbosity_lv, repetitions, outfile, count_time, mem_info);

	//set spikes of neurons in first layer and add their rules
	for(int i=0; i<size; i++){
		snp->set_spikes (i, nums[i]);
		//add_rule (uint nid, uchar e_n, uchar e_i, uchar c, uchar p) 
		snp->add_rule(i, 1, 0, 1, 1,0);	
	}

	int e_n_aux = size;
	//add rules in neurons of 2nd layer
	for(int j=size; j<size*2; j++){
		for(int e_n=size; e_n>=1; e_n--){
			if(e_n == e_n_aux){
				snp->add_rule(j, e_n, 1, e_n, 1,0);
			}else{
				snp->add_rule(j, e_n, 1, e_n, 0,0);
			}
		}
		e_n_aux--;
	}

	//Connect 1st 2nd and 3rd layers
	for(int i=0; i<size; i++){
		for(int j=size; j<size*2; j++){
			snp->add_synapse(i,j);
		}
	}

	e_n_aux = size;
	for(int j=size; j<size*2; j++){
		for(int offset=0; offset<e_n_aux; offset++){
			snp->add_synapse(j,j+size+offset);

		}
		e_n_aux--;
	
	}
	
	snp->compute(); 
	
	delete snp;
}

void simple_snp_with_delays(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info){
	
	//Loading one SNP model
	uint m = 5; //num reglas
	uint n = 3; //num neuronas

	SNP_model *snp = init_alg(alg, n, m);
    snp->set_snpconfig(verbosity_lv, repetitions, outfile, count_time, mem_info);
	
	int C0[3] = {0,1,1};
	for (uint i=0; i<n; i++){
		snp->set_spikes (i, C0[i]);
	}

	//add_rule (uint nid, short e_n, short e_i, short c, short p, ushort d) 
	snp->add_rule(0, 1, 1, 1, 1,0); 
	snp->add_rule(0, 2, 1, 2, 0,0);
	snp->add_rule(1, 1, 1, 1, 1,0); 
	snp->add_rule(1, 1, 1, 1, 1, 1);
	snp->add_rule(2, 1, 1, 1, 1,2);

	snp->add_synapse(0,1);
	snp->add_synapse(1,0);
	snp->add_synapse(0,2);
	snp->add_synapse(2,0);
	
	snp->compute();
	delete snp;

}

