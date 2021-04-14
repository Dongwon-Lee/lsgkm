/* gkmpredict.c
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

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "libsvm_gkm.h"

#define CLOG_MAIN
#include "clog.h"

void print_usage_and_exit()
{
    printf(
            "\n"
            "Program: gkmpredict (lsgkm program for scoring sequences using a trained model)\n"
            "Version: "
            LSGKM_VERSION
            "\n\n"
            "Usage: gkmpredict [options] <test_seqfile> <model_file> <output_file>\n"
            "\n"
            " score test sequences using trained gkm-SVM\n"
            "\n"
            "Arguments:\n"
            " test_seqfile: sequence file for test (fasta format)\n"
            " model_file: output of gkmtrain\n"
            " output_file: name of output file\n"
            "\n"
            "Options:\n"
            " -v <0|1|2|3|4>  set the level of verbosity (default: 2)\n"
            "                   0 -- error msgs only (ERROR)\n"
            "                   1 -- warning msgs (WARN)\n"
            "                   2 -- progress msgs at coarse-grained level (INFO)\n"
            "                   3 -- progress msgs at fine-grained level (DEBUG)\n"
            "                   4 -- progress msgs at finer-grained level (TRACE)\n"
            "-T <1|4|16>      set the number of threads for parallel calculation, 1, 4, or 16\n"
            "                 (default: 1)\n"
            "\n");
    exit(0);
}

static struct svm_model* model;

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

double calculate_score(char *seq)
{
    union svm_data x;
    double score;

    x.d = gkmkernel_new_object(seq, NULL, 0);

    svm_predict_values(model, x, &score);

    gkmkernel_delete_object(x.d);

    return score;
}

void predict(FILE *input, FILE *output)
{
    int iseq = -1;
    char seq[MAX_SEQ_LENGTH];
    char sid[MAX_SEQ_LENGTH];
    int seqlen = 0;
    sid[0] = '\0';
    while (readline(input)) {
        if (line[0] == '>') {
            if (iseq >= 0) {
                double score = calculate_score(seq);
                fprintf(output, "%s\t%g\n",sid, score);
                if ((iseq + 1) % 100 == 0) {
                    clog_info(CLOG(LOGGER_ID), "%d scored", iseq+1);
                }
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

    // last one
    double score = calculate_score(seq);
    fprintf(output, "%s\t%g\n",sid, score);

    clog_info(CLOG(LOGGER_ID), "%d scored", iseq+1);

}


int main(int argc, char **argv)
{
    FILE *input, *output;
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

	int c;
	while ((c = getopt (argc, argv, "v:T:")) != -1) {
		switch (c) {
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
	char *testfile = argv[index++];
	char *modelfile = argv[index++];
	char *outfile = argv[index];

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

    input = fopen(testfile,"r");
    if(input == NULL) {
        clog_error(CLOG(LOGGER_ID),"can't open input file %s", testfile);
        exit(1);
    }

    output = fopen(outfile,"w");
    if(output == NULL) {
        clog_error(CLOG(LOGGER_ID),"can't open output file %s", outfile);
        exit(1);
    }

    clog_info(CLOG(LOGGER_ID), "load model %s", modelfile);
    if((model=svm_load_model(modelfile))==0) {
        clog_error(CLOG(LOGGER_ID),"can't open model file %s", modelfile);
        exit(1);
    }

    max_line_len = 1024;
    line = (char *)malloc(((size_t) max_line_len) * sizeof(char));

    clog_info(CLOG(LOGGER_ID), "write prediction result to %s", outfile);
    predict(input, output);
    svm_free_and_destroy_model(&model);
    free(line);
    fclose(input);
    fclose(output);
    return 0;
}
