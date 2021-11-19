#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <snp_model.hpp>
#include <math.h>
#include "tests.hpp"
#include <getopt.h>

typedef void(*Samples)(int, int, int, char*, bool, bool, int);

void print_usage(char* argv[]){

	printf("SPARSESNP, a project to test sparse matrix representations of SNP on GPUs.\n");
	printf("\nFormat: %s -e[example] -a[algorithm] [OPTIONS]\n",argv[0]);
	printf("Where: \n");
	//printf("\n[i] is the input file describing the SNP to be simulated (check format in help file of repository)\n");
	printf("\n[example] is the example index:\n");
	printf("\t0 = simple SNP\n");
	printf("\t1 = sorting of natural numbers. Give an input size with flag [-n size].\n(i.e [-n 10] will sort an array [10,9...1] in ascending order. Default size value is 50\n");
	printf("\t2 = simple SNP (with delays)\n");
	printf("\t3 = subset sum (with delays)\n");
	printf("\n[algorithm] is the algorithm index\n");
	printf("\t0 = No compression\n");
	printf("\t1 = ELL\n");
	printf("\t2 = OPTIMIZED\n");
	printf("\t3 = cuBLAS (use only with examples that doesn't use delays)\n");
	printf("\t4 = cuSPARSE (use only with examples that doesn't use delays)\n");
	printf("\n[OPTIONS] available (optional):\n");
	printf("\t[-o outfile] = Writes to outfile last configuration computed\n");
	printf("\t[-r repetitions] = Repeat the whole computation \"repetition\" times\n");
	printf("\t[-v level] = Set a level of verbosity\n");
	printf("\t[-t] = Set flag to measure execution time\n");
	printf("\t[-m] = Set flag for memory usage info\n");
	printf("\t[-n] = Give the example the size of the input\n");
	// TODO: Read input file of an snp
}

int main(int argc, char* argv[])
{
	//main args
	int algorithm = -1;
	int example = -1;
	int input_size = -1;

	//option args
	char* outfile = NULL;
	int repetitions = -1;
	int verbosity = 0;
	bool count_time = false;
	bool mem_info = false;
	
	char opt;
	while ((opt = getopt(argc, argv, "e:a:o:r:v:tmn:")) != -1) {
		switch (opt) {
                case 'e':
                	example = atoi(optarg);
					if(example > 3) {
						print_usage(argv);
						printf("\n\nERROR: Selected example does not exist\n");
						return 0;
					}
                  	break;
               	case 'a':
					algorithm = atoi(optarg);
					if(algorithm > 4){
						print_usage(argv);
						printf("\n\nERROR: Selected algorithm does not exist\n");
						return 0;
					}
					break;
			   	case 'o':
					outfile = strdup(optarg);
					break;
			   	case 'r':
				   repetitions = atoi(optarg);
				   break;
			   	case 'v':
				   verbosity = atoi(optarg);
				   break;
				case 't':
					count_time = true;
					break;
				case 'm':
					mem_info = true;
					break;
				case 'n':
					input_size = atoi(optarg);
					break;
			   	default:
                   	print_usage(argv);
                   	exit(0);
        }
           

	}

	if (algorithm == -1 || example == -1) {
		print_usage(argv);
		return 0;
	}

	if((algorithm == GPU_CUBLAS || algorithm == GPU_CUSPARSE )&& example >= 2){
		print_usage(argv);
		printf("\n\nERROR: That algorithm can only be used with examples that doesn't use delays\n");
		return 0;
	}

	Samples samples[] = {&simple_snp, &sort_numbers_snp, &simple_snp_with_delays, &testSubsetSumNonUniformDelays};
	samples[example](algorithm, verbosity, repetitions, outfile, count_time, mem_info, input_size);
	
	//params.debug=1;	
	//while (!vars.halt) {
	//snp_simulator(&params,&vars);
	//}
		
	//free_memory(&params,&vars);
	
	return 0;
}

