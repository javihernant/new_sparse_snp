#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <snp_model.hpp>
#include <math.h>
#include "tests.hpp"
#include <getopt.h>

typedef void(*Samples)(int, int, int, char*, int);

void print_usage(char* argv[]){

	printf("SPARSESNP, a project to test sparse matrix representations of SNP on GPUs.\n");
	printf("\nFormat: %s -e[example] -a[algorithm] [OPTIONS]\n",argv[0]);
	printf("Where: \n");
	//printf("\n[i] is the input file describing the SNP to be simulated (check format in help file of repository)\n");
	printf("\n[example] is the example index:\n");
	printf("\t0 = simple SNP\n");
	printf("\t1 = sorting of natural numbers\n");
	printf("\t2 = simple SNP with delays\n");
	printf("\t3 = subset sum\n");
	printf("\n[algorithm] is the algorithm index\n");
	printf("\t0 = No compression\n");
	printf("\t1 = ELL\n");
	printf("\t2 = OPTIMIZED\n");
	printf("\n [OPTIONS] available (optional):\n");
	printf("\t[-o outfile] = Writes to outfile last configuration computed\n");
	printf("\t[-r repetitions] = Repeat the whole computation \"repetition\" times\n");
	printf("\t[-v level] = Set a level of verbosity\n");
	printf("\t[-t] = Set flag to measure execution time\n");
	// printf("\t1 = GPU lineal algebra CUBLAS\n");
	// printf("\t2 = GPU sparse representation CUSPARSE\n");
	// TODO: Read input file of an snp

}

int main(int argc, char* argv[])
{
	//main args
	int algorithm = -1;
	int example = -1;

	//option args
	char* outfile = NULL;
	int repetitions = 0;
	int verbosity = 0;
	int count_time = 0;
	
	char opt;
	while ((opt = getopt(argc, argv, "e:a:o:r:v:t")) != -1) {
		switch (opt) {
                case 'e':
                	example = atoi(optarg);
					if(example > 3) print_usage(argv);
                  	break;
               	case 'a':
					algorithm = atoi(optarg);

					if(algorithm > 2) print_usage(argv);
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
					count_time = 1;
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

	

	Samples samples[] = {&simple_snp, &sort_numbers_snp, &simple_snp_with_delays};
	samples[example](algorithm, verbosity, repetitions, outfile, count_time);
	
	//params.debug=1;	
	//while (!vars.halt) {
	//snp_simulator(&params,&vars);
	//}
		
	//free_memory(&params,&vars);
	
	return 0;
}
