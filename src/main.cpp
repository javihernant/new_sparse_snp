#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <snp_model.hpp>
#include <math.h>
#include "tests.hpp"
#include <getopt.h>

typedef void(*Samples)(int, int, int, char*);

void print_usage(char* argv[]){

	printf("SPARSESNP, a project to test sparse matrix representations of SNP on GPUs.\n");
	printf("\nFormat: %s -e[example] -a[algorithm] [OPTIONS]\n",argv[0]);
	printf("Where: \n");
	//printf("\n[i] is the input file describing the SNP to be simulated (check format in help file of repository)\n");
	printf("\n[example] is the example index:\n");
	printf("\t0 = simple SNP\n");
	printf("\t1 = sort of natural numbers\n");
	printf("\t2 = subset sum\n");
	printf("\t3 = simple SNP with delays\n");
	printf("\n[algorithm] is the algorithm index\n");
	printf("\t0 = No compression\n");
	printf("\t1 = ELL\n");
	printf("\t2 = OPTIMIZED\n");
	printf("\n (optional) available [OPTIONS]:\n");
	printf("\t [-o outfile] = Enables output to be printed to file. outfile is the name of the file\n");
	printf("\t [-r repetitions] = Repeat computation \"repetition\" times\n");
	printf("\t [-v level] = Set a level of verbosity\n");
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
	int repetitions;
	int verbosity_lv;
	
	char opt;
	while ((opt = getopt(argc, argv, "e:a:o:r:v:")) != -1) {
		switch (opt) {
                case 'e':
                	example = atoi(optarg);
					printf("example:%d\n",example);
					if(example > 3) print_usage(argv);
                  	break;
               	case 'a':
					algorithm = atoi(optarg);
					printf("algorithm:%d\n",example);

					if(algorithm > 2) print_usage(argv);
					break;
			   	case 'o':
					outfile = strdup(optarg);
					break;
			   	case 'r':
				   repetitions = atoi(optarg);
				   break;
			   	case 'v':
				   verbosity_lv = atoi(optarg);
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

	

	Samples samples[] = {&simple_snp};
	samples[example](algorithm, verbosity_lv, repetitions, outfile);
	
	//params.debug=1;	
	//while (!vars.halt) {
	//snp_simulator(&params,&vars);
	//}
		
	//free_memory(&params,&vars);
	
	return 0;
}
