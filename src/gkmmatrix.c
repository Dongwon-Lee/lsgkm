/* gkmmatrix.c
 *
 * Copyright (C) 2015 Dongwon Lee
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

// Notes:
// gkmmatrix is mainly used for debug

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
            "Usage: gkmmatrix [options] <pos_seqfile> <neg_seqfile> <output_kernel>\n"
            "\n"
            " build kernel matrix of gkm-SVM\n"
            "\n"
            "Arguments:\n"
            " pos_seqfile: positive sequence file (fasta format)\n"
            " neg_seqfile: negative sequence file (fasta format)\n"
            " output_kernel: output kernel file\n"
            "\n"
            "Options:\n"
            " -t <0 ~ 5>   set type of kernel function (default: 2)\n"
            "                0 -- gapped-kmer (gkm)\n"
            "                1 -- estimated l-mer with full filter\n"
            "                2 -- estimated l-mer with truncated filter\n"
            "                3 -- truncated filter + radial basis function (RBF)\n"
            "                4 -- truncated filter + positional weights (PW)\n"
            "                5 -- truncated filter + PW + RBF\n"
            " -l <int>     set word length, 3<=l<=12 (default: 10)\n"
            " -k <int>     set number of informative column, k<=l (default: 6)\n"
            " -d <int>     set maximum number of mismatches to consider, d<=4 (default: 3)\n"
            " -g <float>   set gamma for RBF kernel (-t 3 or 5) (default: 1.0)\n"
            " -v <0 ~ 4>   set the level of verbosity (default: 2)\n"
            "                0 -- error msgs only (ERROR)\n"
            "                1 -- warning msgs (WARN)\n"
            "                2 -- progress msgs at coarse-grained level (INFO)\n"
            "                3 -- progress msgs at fine-grained level (DEBUG)\n"
            "                4 -- progress msgs at finer-grained level (TRACE)\n"
            "-T <1|4|16>   set the number of threads for parallel calculation, 1, 4, or 16\n"
            "              (default: 1)\n"
            "\n");

	exit(0);
}

void read_problem(const char *posfile, const char *negfile);
void get_matrix(const char *outkernel);

static struct svm_parameter param;
static struct svm_problem prob;        // set by read_problem

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
        line = (char *) realloc(line,(size_t) max_line_len);
        int len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    
    //remove CR ('\r') or LF ('\n'), whichever comes first
    line[strcspn(line, "\r\n")] = '\0';

    return line;
}

int main(int argc, char** argv)
{
    const char *error_msg;

    int verbosity = 2;
    int nthreads = 1;

    /* Initialize the logger */
    if (clog_init_fd(LOGGER_ID, 1) != 0) {
        fprintf(stderr, "Logger initialization failed.\n");
        return 1;
    }

    clog_set_fmt(LOGGER_ID, LOGGER_FORMAT);
    clog_set_level(LOGGER_ID, CLOG_INFO);

	if(argc == 1) { print_usage_and_exit(); }

    // default values
    param.svm_type = C_SVC;
    param.kernel_type = EST_TRUNC;
    param.L = 10;
    param.k = 6;
    param.d = 3;
    param.gamma = 1.0;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.shrinking = 0;
    param.nr_weight = 0;
    param.weight_label = (int *) malloc(sizeof(int)*1);
    param.weight = (double *) malloc(sizeof(double)*1);
    param.p = 0.1; //not used
    param.probability = 0; //not used
    param.nu = 0.5; //not used

	int c;
	while ((c = getopt (argc, argv, "t:l:k:d:g:v:T:")) != -1) {
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
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 'T':
                nthreads = atoi(optarg);
                break;
			default:
                fprintf(stderr,"Unknown option: -%c\n", c);
                print_usage_and_exit();
		}
	}

    if (argc - optind != 3) {
        fprintf(stderr,"Wrong number of arguments [%d].\n", argc - optind);
        print_usage_and_exit();
    }

	int index = optind;
	char *posfile = argv[index++];
	char *negfile = argv[index++];
	char *outkernel = argv[index];

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
    clog_info(CLOG(LOGGER_ID), "  pos_seqfile = %s", posfile);
    clog_info(CLOG(LOGGER_ID), "  neg_seqfile = %s", negfile);
    clog_info(CLOG(LOGGER_ID), "  output_kernel = %s", outkernel);

    clog_info(CLOG(LOGGER_ID), "Parameters:");
    clog_info(CLOG(LOGGER_ID), "  kernel-type = %d", param.kernel_type);
    clog_info(CLOG(LOGGER_ID), "  L = %d", param.L);
    clog_info(CLOG(LOGGER_ID), "  k = %d", param.k);
    clog_info(CLOG(LOGGER_ID), "  d = %d", param.d);
    if (param.kernel_type == EST_TRUNC_RBF || param.kernel_type == EST_TRUNC_PW_RBF) {
        clog_info(CLOG(LOGGER_ID), "  gamma = %g", param.gamma);
    }

    gkmkernel_init(&param);

    max_line_len = 1024;
    line = (char *) malloc(sizeof(char) * ((size_t) max_line_len));
    read_problem(posfile, negfile);

    error_msg = svm_check_parameter(&prob,&param);
    if(error_msg) {
        clog_error(CLOG(LOGGER_ID), error_msg);
        exit(1);
    }

    get_matrix(outkernel);

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

void read_fasta_file(const char *filename, int label)
{
    FILE *fp = fopen(filename,"r");
    int nseqs = 0;

    clog_info(CLOG(LOGGER_ID), "reading %s", filename);

    if(fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open file");
        exit(1);
    }

    //count the number of sequences for memory allocation
    while(readline(fp)!=NULL) {
        if (line[0] == '>') {
            ++nseqs;
        }
    }
    rewind(fp);

    if (prob.l == 0) {
        prob.y = (double *) malloc (sizeof(double) * ((size_t) nseqs));
        prob.x = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) nseqs));
    } else {
        double *new_prob_y_ptr = 
            (double *) realloc(prob.y, sizeof(double) * ((size_t) (prob.l + nseqs)));
        union svm_data *new_prob_x_ptr = 
            (union svm_data *) realloc(prob.x, sizeof(union svm_data) * ((size_t) (prob.l + nseqs)));

        if (new_prob_y_ptr == NULL || new_prob_x_ptr == NULL) {
            clog_error(CLOG(LOGGER_ID), "error occured while reading sequence file\n");
            exit(1);
        }

        prob.y = new_prob_y_ptr;
        prob.x = new_prob_x_ptr;
    }

    int iseq = -1;
    char seq[MAX_SEQ_LENGTH];
    char sid[MAX_SEQ_LENGTH];
    int seqlen = 0;
    sid[0] = '\0';
    while (readline(fp)) {
        if (line[0] == '>') {
            if (((iseq % 1000) == 0)) {
                clog_info(CLOG(LOGGER_ID), "reading... %d/%d", iseq, nseqs);
            }

            if (iseq >= 0) {
                prob.y[prob.l + iseq] = label;
                prob.x[prob.l + iseq].d = gkmkernel_new_object(seq, sid, prob.l + iseq);
            }
            ++iseq;

            seq[0] = '\0'; //reset sequence
            seqlen = 0;
            char *ptr = strtok(line," \t\r\n");
            if (strlen(ptr) >= MAX_SEQ_LENGTH) {
                clog_error(CLOG(LOGGER_ID), "maximum sequence id length is %d.\n", MAX_SEQ_LENGTH-1);
                exit(1);
            }
            strcpy(sid, ptr+1);
        } else {
            if (seqlen < MAX_SEQ_LENGTH-1) {
                if ((((size_t) seqlen) + strlen(line)) >= MAX_SEQ_LENGTH) {
                    clog_warn(CLOG(LOGGER_ID), "maximum sequence length allowed is %d. The first %d nucleotides of %s will only be used (Note: You can increase the MAX_SEQ_LENGTH parameter in libsvm_gkm.h and recompile).", MAX_SEQ_LENGTH-1, MAX_SEQ_LENGTH-1, sid);
                    int remaining_len = MAX_SEQ_LENGTH - seqlen - 1;
                    line[remaining_len] = '\0';
                }
                strcat(seq, line);
                seqlen += (int) strlen(line);
            }
        }
    }

    //remaining one
    prob.y[prob.l + iseq] = label;
    prob.x[prob.l + iseq].d = gkmkernel_new_object(seq, sid, prob.l + iseq);
    ++iseq;

    clog_info(CLOG(LOGGER_ID), "reading... %d/%d done", iseq, nseqs);

    prob.l += nseqs;
    fclose(fp);
}

void read_problem(const char *posfile, const char*negfile)
{
    prob.l = 0;

    read_fasta_file(posfile, 1);
    read_fasta_file(negfile, -1);
}

void get_matrix(const char *outkernel)
{
    int l = prob.l;
    int i, j;

    FILE *fp = fopen(outkernel, "w");
    if(fp==NULL) {
        clog_error(CLOG(LOGGER_ID), "error occurred while opening output matrix file %s", outkernel);
        return;
    }

    double *kvalue = (double *) malloc(sizeof(double) * ((size_t) l));
    for (i=0;i<l;i++)
    {
        gkmkernel_kernelfunc_batch(prob.x[i].d, prob.x, i+1, kvalue);
        for(j=0;j<=i;j++)
        {
            fprintf(fp, "%g ", kvalue[j]);
        }
        fprintf(fp, "\n");
    }
    free(kvalue);
}
