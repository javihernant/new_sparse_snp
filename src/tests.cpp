#include "tests.hpp"
#include <iostream>
#include <new>
#include <time.h> 
#include "error_check.hpp"


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


void simple_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size){
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

void sort_numbers_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size){
	int size = 50; //adjust to choose how many numbers are sorted
	if(input_size>0){
		size = input_size;
	}
	printf("Input size: %d\n",size);

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

void simple_snp_with_delays(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size){
	
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

void testSubsetSumNonUniformDelays(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size){

	int v_size = input_size > 0 ? input_size : 50;
	printf("Input size: %d\n",v_size);
	printf("Execution will repeat %d times\n",repetitions);

	int v[v_size];
	int S=0;
	int seed = 28;
	std::srand(seed);
	for(int i=0; i<v_size; i++){
		v[i] = ( 1+ std::rand() % ( 50 + 1 ) ); //generates a number in the range 0-50

		if((std::rand() % 100)<20){ //20% of the time the element will be chosen for the total sum (S)
			S+=v[i];
		}
	}

	int sum_of_v = 0;
	for(int i=0; i<v_size; i++){
		sum_of_v += v[i];
	}

	uint n = v_size*2 + sum_of_v +3; //num neuronas
	uint m = v_size*2*2 + sum_of_v + 2; //num reglas

	int initial_reps = repetitions;

	float time_ms;
    cudaEvent_t start, stop;  
	if(count_time){
        cuda_check( cudaEventCreate(&start) );
        cuda_check( cudaEventCreate(&stop) );
        cuda_check( cudaEventRecord(start, 0) );
    }

	std::srand(time(NULL));	//from now on, rules will be selected randomly in each repetition
	while(repetitions--){
		if(verbosity_lv>=1){
			printf("test repetition #%d\n",initial_reps-repetitions);
		}
		SNP_model *snp = init_alg(alg, n, m);
		snp->set_snpconfig(verbosity_lv, initial_reps, outfile, count_time, mem_info);

		for (int i=0; i<v_size+1; i++){
			snp->set_spikes (i, 1);
		}

		snp->add_rule(0, 1, 1, 1, 1,0);
		for (int i=1; i<=v_size; i++){
			if((std::rand() % 2)==0){
				snp->add_rule(i, 1, 1, 1, 1,0);
				snp->add_rule(i, 1, 1, 1, 1,1);
			}else{
				snp->add_rule(i, 1, 1, 1, 1,1);
				snp->add_rule(i, 1, 1, 1, 1,0);
			}	
		}
		
		for (int i=v_size+1; i<=v_size*2; i++){
			snp->add_rule(i, 2, 1, 2, 1,0);
			snp->add_rule(i, 1, 1, 1, 0,0);
		}
		int neuron = v_size*2+1;
		
		for (int i=0; i<v_size; i++){
			
			for(int offset=0; offset<v[i]; offset++){
				snp->add_rule(neuron+offset, 1, 1, 1, 1,0);
			}
			neuron+=v[i];
		}
		snp->add_rule(neuron, S, 1, S, 1,0);
		
		//Adding synapses
		for (int i=v_size+1; i<=v_size*2; i++){
			snp->add_synapse(0,i);
		}

		for (int i=1; i<=v_size; i++){
			snp->add_synapse(i,v_size*2+i);
		}

		int j_n = v_size*2 + 1;
		for (int i=0; i<v_size; i++){
			int i_n = v_size+1+ i;
			for(int c=0; c<v[i]; c++){
				snp->add_synapse(i_n,j_n);
				j_n++;
			}
				
		}

		//connecting to output neuron
		for(int i_n=v_size*2+1; i_n<j_n; i_n++){
			snp->add_synapse(i_n,j_n); 

		}
		//connecting out_neuron to enviroment neuron
		snp->add_synapse(j_n, j_n+1);
		
		snp->compute(4); //4 steps at most, 2 at minimum
	}

	if(count_time){
        cuda_check( cudaEventRecord(stop, 0) );
        cuda_check( cudaEventSynchronize(stop) );
        cuda_check( cudaEventElapsedTime(&time_ms, start, stop) );

        printf("Execution time: %3.1f ms\n", time_ms);
    }

	if(mem_info){
        size_t free_bytes;
        size_t total_bytes;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
        double used_mem = (total_bytes - free_bytes)/1024/1024;
        printf("Used memory: %f MB\n", used_mem);
    }




}	

	
	
	

