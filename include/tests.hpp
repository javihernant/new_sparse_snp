#ifndef _TESTS_HPP_
#define _TESTS_HPP_

#include "snp_model.hpp"
#include "snp_static.hpp"

SNP_model* init_alg(int algorithm, int neurons, int rules);
void simple_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info);
void sort_numbers_snp(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info);
void simple_snp_with_delays(int alg, int verbosity_lv, int repetitions, char* outfile, bool count_time, bool mem_info);


#endif