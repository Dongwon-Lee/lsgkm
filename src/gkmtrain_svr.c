/* gkmtrain_svr.c
 *
 * Copyright (C) 2021 Dongwon Lee
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "libsvm_gkm.h"

#define CLOG_MAIN
#include "clog.h"

///////////////////////////////////////////////////////////////////////////////
void print_usage_and_exit()
{
    printf(
            "\n"
            "Program: gkmtrain-svr (lsgkm program for SVR model training)\n"
            "Version: "
            LSGKM_VERSION
            "\n\n"
            "Usage: gkmtrain [options] <datafile> <outprefix>\n"
            "\n"
            " train support vector regression (SVR) using gkm-kenrel and libSVM\n"
            "\n"
            "Arguments:\n"
            " datafile: tab-delimited data file. The 1st column is sequence and\n"
            "           the 2nd column is score.\n"
            " outprefix: prefix of output file(s) <outprefix>.model.txt or\n"
            "            <outprefix>.cvpred.txt\n"
            "\n"
            "Options:\n"
            " -t <0 ~ 5>   set kernel function (default: 2 gkm)\n"
            "              NOTE: RBF kernels (3 and 5) work best with -c 10 -g 2\n"
            "                0 -- gapped-kmer\n"
            "                1 -- estimated l-mer with full filter\n"
            "                2 -- estimated l-mer with truncated filter (gkm)\n"
            "                3 -- gkm + RBF (gkmrbf)\n"
            "                4 -- gkm + center weighted (wgkm)\n"
            "                     [weight = max(M, floor(M*exp(-ln(2)*D/H)+1))]\n"
            "                5 -- gkm + center weighted + RBF (wgkmrbf)\n"
            " -l <int>     set word length, 3<=l<=12 (default: 11)\n"
            " -k <int>     set number of informative column, k<=l (default: 7)\n"
            " -d <int>     set maximum number of mismatches to consider, d<=4 (default: 3)\n"
            " -g <float>   set gamma for RBF kernel. -t 3 or 5 only (default: 1.0)\n"
            " -M <int>     set the initial value (M) of the exponential decay function\n"
            "              for wgkm-kernels. max=255, -t 4 or 5 only (default: 50)\n"
            " -H <float>   set the half-life parameter (H) that is the distance (D) required\n"
            "              to fall to half of its initial value in the exponential decay\n"
            "              function for wgkm-kernels. -t 4 or 5 only (default: 50)\n"
            " -R           if set, reverse-complement is not considered as the same feature\n"
            " -c <float>   set the regularization parameter C (default: 0.1)\n"
            " -p <float>   set the epsilon parameter in loss function of SVR (default: 0.1)\n"
            " -e <float>   set the precision parameter (default: 0.001)\n"
            " -m <float>   set cache memory size in MB (default: 100.0)\n"
            "              NOTE: Large cache signifcantly reduces runtime. >4Gb is recommended\n" 
            " -s           if set, use the shrinking heuristics\n"
            " -x <int>     set N-fold cross validation mode (default: no cross validation)\n"
            " -i <int>     run i-th cross validation only 1<=i<=ncv (default: all)\n"
            " -r <int>     set random seed for shuffling in cross validation mode (default: 1)\n"
            " -v <0 ~ 4>   set the level of verbosity (default: 2)\n"
            "                0 -- error msgs only (ERROR)\n"
            "                1 -- warning msgs (WARN)\n"
            "                2 -- progress msgs at coarse-grained level (INFO)\n"
            "                3 -- progress msgs at fine-grained level (DEBUG)\n"
            "                4 -- progress msgs at finer-grained level (TRACE)\n"
            "-T <1|4|16>   set the number of threads for parallel calculation, 1, 4, or 16\n"
            "                 (default: 1)\n"
            "\n");

	exit(0);
}

static struct svm_parameter param;
static struct svm_problem prob;        // set by read_problem
static struct svm_model *model;
static int cross_validation;
static int icv;
static int nr_fold;

static char *line = NULL;
static int max_line_len;

// this function was copied from libsvm & slightly modified 
static char* readline(FILE *input)
{
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line, (size_t) max_line_len);
        int len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    
    //remove CR ('\r') or LF ('\n'), whichever comes first
    line[strcspn(line, "\r\n")] = '\0';

    return line;
}

static void read_training_data_file(const char *filename)
{
    FILE *fp = fopen(filename,"r");

    if(fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open input file %s", filename);
        exit(1);
    }

    int iseq = -1;
    char seq[MAX_SEQ_LENGTH];
    double expr = 0;
    while (readline(fp)) {
        if (iseq >= prob.l) {
            clog_error(CLOG(LOGGER_ID), "error occured while reading training data file (%d > %d).\n", iseq, prob.l);
            exit(1);
        }

        iseq++;

        char *ptr = strtok(line," \t\r\n");

        // get sequence
        if (ptr != NULL) {
            if (strlen(ptr) >= MAX_SEQ_LENGTH) {
                clog_error(CLOG(LOGGER_ID), "maximum sequence length is %d.\n", MAX_SEQ_LENGTH-1);
                exit(1);
            }
            strcpy(seq, ptr);
        } else {
            clog_error(CLOG(LOGGER_ID), "illigal format of training data file.\n");
            exit(1);
        }

        // get relative normalized expression 
        ptr = strtok (NULL, " \t\r\n");
        if (ptr != NULL) {
            char *end;
            expr = strtod(ptr, &end);
        } else {
            clog_error(CLOG(LOGGER_ID), "illigal format of training data file.\n");
            exit(1);
        }

        //printf("%s, %f\n", seq, expr);
        prob.y[iseq] = expr;
        prob.x[iseq].d = gkmkernel_new_object(seq, seq, iseq);
    }

    fclose(fp);
}

static int count_lines(const char *filename)
{
    FILE *fp = fopen(filename,"r");
    int nseqs = 0;

    if(fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open input file %s", filename);
        exit(1);
    }

    //count the number of lines for memory allocation
    while(readline(fp)!=NULL) {
        ++nseqs;
    }
    fclose(fp);
    
    return nseqs;
}

static void read_problem(const char *tdfile)
{
    int n = count_lines(tdfile);

    prob.l = n;

    prob.y = (double *) malloc (sizeof(double) * ((size_t) n));
    prob.x = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) n));

    clog_info(CLOG(LOGGER_ID), "reading %d tags from %s", n, tdfile);
    read_training_data_file(tdfile);
}

static void do_cross_validation(const char *filename)
{
    double *target = (double *) malloc(sizeof(double) * ((size_t) prob.l));

    svm_cross_validation(&prob,&param,nr_fold,icv,target,filename);

    free(target);
}

int main(int argc, char** argv)
{
    char model_file_name[1024];
    char cvpred_file_name[1024];
    const char *error_msg;

    int verbosity = 2;
    int nthreads = 1;
	int rseed = 1;
    int tmpM;

    /* Initialize the logger */
    if (clog_init_fd(LOGGER_ID, 1) != 0) {
        fprintf(stderr, "Logger initialization failed.\n");
        return 1;
    }

    clog_set_fmt(LOGGER_ID, LOGGER_FORMAT);
    clog_set_level(LOGGER_ID, CLOG_INFO);

	if(argc == 1) { print_usage_and_exit(); }

    // default values
    param.svm_type = EPSILON_SVR;
    param.kernel_type = EST_TRUNC;
    param.L = 11;
    param.k = 7;
    param.d = 3;
    param.M = 50;
    param.H = 50;
    param.gamma = 1.0;
    param.cache_size = 100;
    param.C = 0.1;
    param.p = 0.1;
    param.eps = 0.001;

    param.shrinking = 0;
    param.nr_weight = 0;
    param.weight_label = (int *) malloc(sizeof(int)*1);
    param.weight = (double *) malloc(sizeof(double)*1);
    param.norc = 0;
    param.probability = 0; //not used
    param.nu = 0.5; //not used
    cross_validation = 0;
    icv = 0;

	int c;
	while ((c = getopt (argc, argv, "t:l:k:d:g:M:H:c:p:e:m:x:i:r:sv:T:R")) != -1) {
		switch (c) {
            case 't':
                param.kernel_type = atoi(optarg);
                break;
            case 'l':
                param.L = atoi(optarg);
                break;
            case 'k':
                param.k = atoi(optarg);
                break;
            case 'd':
                param.d = atoi(optarg);
                break;
            case 'g':
                param.gamma = atof(optarg);
                break;
            case 'M':
                tmpM = atoi(optarg);
                if (tmpM > 255) {
                    clog_warn(CLOG(LOGGER_ID), "maximum M is 255. M is set to default value [%d].", param.M);
                } else {
                    param.M = (uint8_t) tmpM;
                }
                break;
            case 'H':
                param.H = atof(optarg);
                break;
			case 'c':
				param.C = atof(optarg);
				break;
			case 'p':
				param.p = atof(optarg);
				break;
			case 'e':
				param.eps = atof(optarg);
				break;
			case 'w':
                param.nr_weight = 1;
                param.weight_label[0] = 1;
                param.weight[0] = atof(optarg);
				break;
            case 'm':
                param.cache_size = atof(optarg);
                break;
			case 'x':
                cross_validation = 1;
                nr_fold = atoi(optarg);
                if(nr_fold < 2) {
                    fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                    print_usage_and_exit();
                }
				break;
			case 'i':
				icv = atoi(optarg);
				break;
			case 'r':
				rseed = atoi(optarg);
				break;
            case 's':
                param.shrinking = 1;
                break;
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 'T':
                nthreads = atoi(optarg);
                break;
            case 'R':
                param.norc = 1;
                break;
			default:
                fprintf(stderr,"Unknown option: -%c\n", c);
                print_usage_and_exit();
		}
	}

    if (argc - optind != 2) {
        fprintf(stderr,"Wrong number of arguments [%d].\n", argc - optind);
        print_usage_and_exit();
    }

	int index = optind;
	char *datafile = argv[index++];
	char *outprefix = argv[index];

    switch(verbosity) 
    {
        case 0:
            clog_set_level(LOGGER_ID, CLOG_ERROR);
            break;
        case 1:
            clog_set_level(LOGGER_ID, CLOG_WARN);
            break;
        case 2:
            clog_set_level(LOGGER_ID, CLOG_INFO);
            break;
        case 3:
            clog_set_level(LOGGER_ID, CLOG_DEBUG);
            break;
        case 4:
            clog_set_level(LOGGER_ID, CLOG_TRACE);
            break;
        default:
            fprintf(stderr, "Unknown verbosity: %d\n", verbosity);
            print_usage_and_exit();
    }

    gkmkernel_set_num_threads(nthreads);

    clog_info(CLOG(LOGGER_ID), "Arguments:");
    clog_info(CLOG(LOGGER_ID), "  datafile = %s", datafile);
    clog_info(CLOG(LOGGER_ID), "  outprefix = %s", outprefix);

    clog_info(CLOG(LOGGER_ID), "Parameters:");
    clog_info(CLOG(LOGGER_ID), "  kernel-type = %d", param.kernel_type);
    clog_info(CLOG(LOGGER_ID), "  L = %d", param.L);
    clog_info(CLOG(LOGGER_ID), "  k = %d", param.k);
    clog_info(CLOG(LOGGER_ID), "  d = %d", param.d);
    if (param.kernel_type == EST_TRUNC_RBF || param.kernel_type == EST_TRUNC_PW_RBF) {
    clog_info(CLOG(LOGGER_ID), "  gamma = %g", param.gamma);
    }
    if (param.kernel_type == EST_TRUNC_PW || param.kernel_type == EST_TRUNC_PW_RBF) {
    clog_info(CLOG(LOGGER_ID), "  M = %d", param.M);
    clog_info(CLOG(LOGGER_ID), "  H = %g", param.H);
    }
    clog_info(CLOG(LOGGER_ID), "  C = %g", param.C);
    if (param.nr_weight == 1) {
    clog_info(CLOG(LOGGER_ID), "  w = %g", param.weight[0]);
    }
    clog_info(CLOG(LOGGER_ID), "  eps = %g", param.eps);
    clog_info(CLOG(LOGGER_ID), "  shrinking = %s", param.shrinking ? "yes" : "no");
    clog_info(CLOG(LOGGER_ID), "  reverse-complement feature = %s", param.norc ? "no" : "yes");

    sprintf(model_file_name,"%s.model.txt", outprefix);

    if (cross_validation) {
        srand((unsigned int) rseed);
        clog_info(CLOG(LOGGER_ID), "random seed is set to %d", rseed);

        if (icv) {
            /* save CV results to this file if -x option is set */
            sprintf(cvpred_file_name,"%s.cvpred.%d.txt", outprefix, icv); 
        } else {
            sprintf(cvpred_file_name,"%s.cvpred.txt", outprefix); 
        }
    } 

    gkmkernel_init(&param);

    max_line_len = 1024;
    line = (char *) malloc(sizeof(char) * ((size_t) max_line_len));
    read_problem(datafile);

    error_msg = svm_check_parameter(&prob,&param);
    if(error_msg) {
        clog_error(CLOG(LOGGER_ID), error_msg);
        exit(1);
    }

    if(cross_validation) {
        do_cross_validation(cvpred_file_name);
    }
    else {
        model = svm_train(&prob,&param);

        clog_info(CLOG(LOGGER_ID), "save SVM model to %s", model_file_name);

        if(svm_save_model(model_file_name,model)) {
            clog_error(CLOG(LOGGER_ID), "can't save model to file %s", model_file_name);
            exit(1);
        }
        svm_free_and_destroy_model(&model);
    }

    int i;
    for (i=0; i<prob.l; i++) {
        gkmkernel_delete_object(prob.x[i].d);
    }

    svm_destroy_param(&param);
    free(prob.y);
    free(prob.x);
    free(line);

	return 0;
}
