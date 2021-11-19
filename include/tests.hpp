#ifndef _TESTS_HPP_
#define _TESTS_HPP_

#include "snp_model.hpp"
#include "snp_static.hpp"

// Algorithms
#define NO_COMPRESSION		0
#define ELL 				1
#define OPTIMIZED			2
#define GPU_CUBLAS 			3
#define GPU_CUSPARSE	 	4

SNP_model* init_alg(int algorithm, int neurons, int rules);
void simple_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size);
void sort_numbers_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size);
void simple_snp_with_delays(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size);
void testSubsetSumNonUniformDelays(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info, int input_size);

#endif