/* libsvm_gkm.c
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include "libsvm_gkm.h"
#include "clog.h"

#define MAX_MM 12

//XXX: only works when MAX_ALPHABET_SIZE = 4
#define LEAF_COUNT(a) (1<<(2*a))  
#define NODE_COUNT(a) ((1<<(2*a))-1)/(MAX_ALPHABET_SIZE-1); // (x^n-1)/(x-1) = 1 + x + x^2 + ... x^(n-1)

static struct svm_parameter *g_param = NULL;
static int g_param_nthreads = 1;

//static double g_param_lambda = 1.0;

/* g_weights are automatically determined based on the g_param->L, g_param->k, and g_param->d */
static double g_weights[MAX_MM] = {0.0};

static KmerTree *g_kmertree = NULL;

static KmerTree *g_prob_kmertree = NULL;
static union svm_data *g_prob_svm_data = NULL;
static int g_prob_num = 0;
static int *g_prob_gkmkernel_index = NULL;
static int *g_prob_libsvm_index = NULL;

static KmerTree *g_sv_kmertree = NULL;
static union svm_data *g_sv_svm_data = NULL;
static int g_sv_num = 0;
static int **g_sv_mmprofile = NULL;

static KmerTreeCoef *g_sv_kmertreecoef = NULL;

typedef struct _BaseMismatchCount {
    uint8_t *bid;
    uint8_t wt;
    int mmcnt;
} BaseMismatchCount;

typedef struct _kmertree_dfs_pthread_t {
    KmerTree *tree;
    int start_depth;
    int start_node_index;
    BaseMismatchCount matching_bases[MAX_SEQ_LENGTH];
    int num_matching_bases;
    int **mmprofile;
    int last_seqid;
} kmertree_dfs_pthread_t;

typedef struct _kmertreecoef_dfs_pthread_t{
    int start_depth;
    int start_node_index;
    BaseMismatchCount matching_bases[MAX_SEQ_LENGTH];
    int num_matching_bases;
    double result;
} kmertreecoef_dfs_pthread_t;

static time_t diff_ms(struct timeval t1, struct timeval t2)
{
    return (t1.tv_sec - t2.tv_sec)*1000 + (t1.tv_usec - t2.tv_usec)/1000;
}

/********************************************
 * various weight calculations from gkmsvm  *
 ********************************************/
static double dCombinations(int n, int r)
{
    if (r<0) return 0; 
    if (n<0) return dCombinations(r-n-1, r)*((r%2==0)?1:-1); 
    if (n<r) return 0; 
    if ((n==0)&&(r==0)) return 1.0; 

    int i,j; 

    double *nn,*no, *h; 
    nn = (double *) malloc (sizeof(double) * ((size_t) (r+1))); 
    no = (double *) malloc (sizeof(double) * ((size_t) (r+1)));

    for(i=0;i<=r;i++)
    {
        nn[i]=no[i]=0; 
    }
    nn[0]=no[0]=1;
    
    for(i=1;i<=n;i++)
    {
        h = no; no = nn; nn=h; 
        for(j=1;j<=r;j++)
        {
            nn[j] = no[j]+no[j-1]; 
        }
    }
    double res = nn[r]; 
    free(nn);
    free(no);

    return res; 
}

static void calc_gkm_kernel_lmerest_wt(int truncated)
{
    int b = MAX_ALPHABET_SIZE;
    int L = g_param->L;
    int K = g_param->k;
    double *res = g_weights;
    double **wL = (double **) malloc(sizeof(double*) * ((size_t) (K+1))); 
    double **wLp =(double **) malloc(sizeof(double*) * ((size_t) (K+1)));
    double *wm = (double *) malloc(sizeof(double) * ((size_t) (K+1))); 
    double *kernel = (double *) malloc(sizeof(double) * ((size_t) (L+1))); 
    double *kernelTr = (double *) malloc(sizeof(double) * ((size_t) (L+1))); 
    double **hv; 
    int i,j; 
    int iL, iK, jM;
    int m;

    /* 1. calculate wm */
    for (i=0; i<=K; i++) {
        wL[i]= (double *) malloc(sizeof(double) * ((size_t) (K+1))); 
        wLp[i]= (double *) malloc(sizeof(double) * ((size_t) (K+1)));

        for (j=0; j<=K; j++) {
            wL[i][j] = wLp[i][j] = 1.0; 
        }
    }

    for (iL=1; iL<=L; iL++) {
        for (iK=1; iK<=K; iK++) {
            wL[iK][0] = wLp[iK][0] + (b-1)* wLp[iK-1][0]; 

            for (jM=1; jM<=iK; jM++) {
                wL[iK][jM] = (wL[iK-1][jM-1] * (iK-iL))/iK;  
            }
        }

        hv = wLp; wLp=wL; wL=hv; 
    }

    double nnorm = dCombinations(L,K)*pow(b,1.0*L); 

    for (i=0; i<=K; i++) {
        wm[i] = wLp[K][i]/nnorm; 
    }

    /* 2. calculate kernel */
    for (m=0; m<=L; m++) {
        int ub = (m < K) ? m : K;
        kernel[m]=0;
        for (i=0; i<=ub; i++) {
            kernel[m]+=wm[i]*dCombinations(L-m,K-i)*dCombinations(m,i);
        }
    }

    int hn=1; 
    for(i=0;i<=L;i++) {
        if (kernel[i] < 1e-50) hn=0; 
        if (hn) {
            kernelTr[i]=kernel[i]; 
        } else {
            kernelTr[i]=0.0;
        }       
    }

    /* 3. calculate wt */
    for (m=0; m<=L; m++) {
        int m1, m2, t;
        double w = 0;
        for (m1=0; m1<=L; m1++) {
            for (m2=0; m2<=L; m2++) {
                for (t=0; t<=L; t++) {
                    int r= m1+m2-2*t-L+m; 
                    if ((t<=m)&&((m1-t)<=(L-m))&&(r<=(m1-t))&&(r>=0)) {
                        double cc = dCombinations(m,t)*dCombinations(L-m,m1-t)*dCombinations(m1-t,r)*pow(b-1, 1.0*t)*pow(b-2, 1.0*r); 
                        if (truncated != 0) {
                            w += cc*kernelTr[m1]*kernelTr[m2]; 
                        } else {
                            w += cc*kernel[m1]*kernel[m2]; 
                        }
                    }
                }
            }
        }

        res[L-m] = w;
    }

    for (i=0; i<=K; i++) {
        free(wL[i]);
        free(wLp[i]);
    }
    free(wL);
    free(wLp);
    free(wm);
    free(kernel);
    free(kernelTr);
}

static void calc_gkm_kernel_wt()
{
    /* corresponding to h[m] */
    int i;
    int L = g_param->L;
    int K = g_param->k;
    double *res = g_weights;

    for (i=0; i<=L; i++) {
        if ((L-i) >= K) {
            res[i] = dCombinations(L-i,K); 
        }
    }
}

/* legacy codes.. 
static void calc_wildcard_kernel_wt() 
{
    // weights corresponding to wildcard kernel of LK2004
    int L = g_param->L;
    int M = g_param_M;
    double lambda = g_param_lambda;
    double *res = g_weights;
    int m;

    for(m=0; m<=L; m++){
        double w = 0; 
        int k;
        for(k=(L-M); k<=L; k++){
            if (L-m >= k){
                w += pow(lambda,1.0*(L-k)) * dCombinations(L-m, k);
            }
        }
        res[m]=w;
    }
}

static void calc_mismatch_kernel_wt() 
{
    // weights corresponding to mismatch kernel of Leslie et al. 2002
    int b = MAX_ALPHABET_SIZE;
    int L = g_param->L;
    int M = g_param_M;
    double *res = g_weights;
    int m;

    for(m=0; m<=L; m++){
        double w = 0; 
        int m1, m2, t;
        for(m1=0; m1<=M; m1++){
            for(m2=0; m2<=M; m2++){
                for(t=0; t<=M; t++){
                    int r = m2+m1-m-2*t; 
                    w += dCombinations(L-m,t)*pow(b-1, 1.0*t)*dCombinations(m,r)*pow(b-2,1.0*r)* dCombinations(m-r, m1-t-r);
                }
            }
        }
        res[m]=w;
    }
}
*/


/************************
 * kmertree functions *
 ************************/
static void kmertree_init(KmerTree *tree, int kmerlen)
{
    tree->depth = kmerlen;
    tree->node_count = NODE_COUNT(kmerlen);
    tree->node = (int *) calloc((size_t) tree->node_count, sizeof(int));

    tree->leaf_count = LEAF_COUNT(kmerlen);
    tree->leaf = (KmerTreeLeaf *) calloc((size_t) tree->leaf_count, sizeof(KmerTreeLeaf));
}

static void kmertree_destroy(KmerTree *tree)
{
    if (tree) {
        if (tree->node) free(tree->node);
        if (tree->leaf) {
            int i;
            for(i=0; i<tree->leaf_count; i++) {
                if (tree->leaf[i].data) free(tree->leaf[i].data);
            }
            free(tree->leaf);
        }
        free(tree);
    }
}

static void kmertree_add_sequence(const KmerTree *tree, int seqid, const gkm_data *d)
{
    int i, j, k;
    uint8_t *seqs[2] = {d->seq, d->seq_rc};
    uint8_t *wts[2] = {d->wt, d->wt_rc};

    for (k=0; k<2; k++) {
        uint8_t *seq = seqs[k];
        uint8_t *wt = wts[k];

        for (j=0; j<(d->seqlen - tree->depth + 1); j++) {
            int node_index = 0;
            int found = 0;
            for (i=0; i<tree->depth; i++) {
                tree->node[node_index]++;
                node_index = (node_index*MAX_ALPHABET_SIZE) + seq[i+j];
            }

            // add the sequence id to the corresponding leaf node
            KmerTreeLeaf *leaf = tree->leaf + node_index - tree->node_count;
            if (leaf->capacity == 0) {
                // initialize stack
                leaf->count = 0;
                leaf->capacity = 1;
                leaf->data = (KmerTreeLeafData *) malloc(sizeof(KmerTreeLeafData)*1);
            } else if (leaf->count == leaf->capacity) {
                // expand stack
                KmerTreeLeafData *newdata = 
                        (KmerTreeLeafData *) malloc(sizeof(KmerTreeLeafData)*((size_t) leaf->capacity)*2);
                int i;
                for (i=0; i<leaf->count; i++) {
                    newdata[i].seqid = leaf->data[i].seqid;
                    newdata[i].wt = leaf->data[i].wt;
                }
                free(leaf->data);
                leaf->capacity *= 2;
                leaf->data = newdata;
            }
            for (i=0; i<leaf->count; i++) {
                if(leaf->data[i].seqid == seqid) {
                    leaf->data[i].wt += (int) wt[j];
                    found = 1;
                    break;
                }
            }
            if (found == 0) {
                leaf->data[leaf->count].seqid = seqid;
                leaf->data[leaf->count].wt = wt[j];
                leaf->count++;
            }
        }
    }
}

/* not used
static void kmertree_delete_sequence(const KmerTree *tree, int seqid, const gkm_data *d)
{
    int i, j, k;
    uint8_t *seqs[2] = {d->seq, d->seq_rc};

    for (k=0; k<2; k++) {
        uint8_t *seq = seqs[k];
        for (j=0; j<(d->seqlen - tree->depth + 1); j++) {
            int node_index = 0;
            for (i=0; i<tree->depth; i++) {
                tree->node[node_index]++;
                node_index = (node_index*MAX_ALPHABET_SIZE) + seq[i+j];
            }

            // set the wt zero from the corresponding leaf node 
            // by scanning the stack
            KmerTreeLeaf *leaf = tree->leaf + node_index - tree->node_count;

            for (i=0; i<leaf->count; i++) {
                if (leaf->data[i].seqid == seqid) {
                    leaf->data[i].wt = 0; //reset
                }
            }
        }
    }
}
*/

static void kmertree_dfs(const KmerTree *tree, const int last_seqid, const int depth, const int curr_node_index, const BaseMismatchCount *curr_matching_bases, const int curr_num_matching_bases, int **mmprof)
{
    int i, j;
    int bid;

    const int d = g_param->d; //for small speed-up

    if (depth == tree->depth - 1) {
        KmerTreeLeaf *leaf = tree->leaf + (curr_node_index*MAX_ALPHABET_SIZE) - tree->node_count;
        for (bid=1; bid<=MAX_ALPHABET_SIZE; bid++) {
            leaf++;
            if (leaf->count > 0) {
                for (j=0; j<curr_num_matching_bases; j++) {
                    const uint8_t currbase = *curr_matching_bases[j].bid;
                    const uint8_t currbase_wt = curr_matching_bases[j].wt;
                    const int currbase_mmcnt = curr_matching_bases[j].mmcnt;
                    if (currbase == bid) {
                        // matching
                        const int leaf_cnt = leaf->count;
                        const KmerTreeLeafData *data = leaf->data;
                        int *mmprof_mmcnt = mmprof[currbase_mmcnt];
                        for (i=0; i<leaf_cnt; i++) { 
                            if (data[i].seqid < last_seqid) {
                                mmprof_mmcnt[data[i].seqid] += (data[i].wt*currbase_wt); 
                            }
                        }
                    } else if (currbase_mmcnt < d) {
                        // non-matching
                        const int leaf_cnt = leaf->count;
                        const KmerTreeLeafData *data = leaf->data;
                        int *mmprof_mmcnt = mmprof[currbase_mmcnt+1];
                        for (i=0; i<leaf_cnt; i++) { 
                            if (data[i].seqid < last_seqid) {
                                mmprof_mmcnt[data[i].seqid] += (data[i].wt*currbase_wt); 
                            }
                        }
                    }
                }
            }
        }
    } else {
        int daughter_node_index = (curr_node_index*MAX_ALPHABET_SIZE);
        for (bid=1; bid<=MAX_ALPHABET_SIZE; bid++) {
            daughter_node_index++;
            if (tree->node[daughter_node_index] > 0) {
                BaseMismatchCount next_matching_bases[MAX_SEQ_LENGTH];
                int next_num_matching_bases = 0;

                for (j=0; j<curr_num_matching_bases; j++) {
                    uint8_t *currbase_ptr = curr_matching_bases[j].bid;
                    int currbase_mmcnt = curr_matching_bases[j].mmcnt;
                    if (*currbase_ptr == bid) {
                        // matching
                        next_matching_bases[next_num_matching_bases].bid = currbase_ptr+1;
                        next_matching_bases[next_num_matching_bases].wt = curr_matching_bases[j].wt;
                        next_matching_bases[next_num_matching_bases].mmcnt = currbase_mmcnt;
                        next_num_matching_bases++;
                    } else if (currbase_mmcnt < d) {
                        // non-matching
                        next_matching_bases[next_num_matching_bases].bid = currbase_ptr+1;
                        next_matching_bases[next_num_matching_bases].wt = curr_matching_bases[j].wt;
                        next_matching_bases[next_num_matching_bases].mmcnt = currbase_mmcnt+1;
                        next_num_matching_bases++;
                    }
                }

                if (next_num_matching_bases > 0) {
                    kmertree_dfs(tree, last_seqid, depth+1, daughter_node_index, next_matching_bases, next_num_matching_bases, mmprof);
                } 
            }
        }
    }
}

static void kmertree_cleanup(const KmerTree *tree, int depth, int curr_node_index)
{
    int bid;

    if (depth == tree->depth - 1) {
        for (bid=1; bid<=MAX_ALPHABET_SIZE; bid++) {
            KmerTreeLeaf *leaf = tree->leaf + (curr_node_index*MAX_ALPHABET_SIZE) + bid - tree->node_count;
            if (leaf->count > 0) {
                // empty the stack
                leaf->count = 0;
            }
        }
    } else {
        for (bid=1; bid<=MAX_ALPHABET_SIZE; bid++) {
            int daughter_node_index = (curr_node_index*MAX_ALPHABET_SIZE) + bid;
            if (tree->node[daughter_node_index] > 0) {
                kmertree_cleanup(tree, depth+1, daughter_node_index);
            }
        }
    }

    tree->node[curr_node_index] = 0; //reset the reference count
}

static void kmertree_dfs_pthread_init_par4(const gkm_data *da, const int last_index, KmerTree *tree, kmertree_dfs_pthread_t *td)
{
    int i, j, k;

    //process the first level and initialize thread input & output variables
    for (i=0; i<MAX_ALPHABET_SIZE; i++) {
        //input
        int bid = i + 1;
        td[i].tree = tree;
        td[i].start_depth = 1;
        td[i].start_node_index = bid;
        td[i].num_matching_bases = 0;
        td[i].last_seqid = last_index;

        uint8_t *seq = da->seq;
        uint8_t *wt = da->wt;
        for (j=0; j<da->seqlen - g_param->L + 1; j++) {
            int mmcnt = 0;
            uint8_t *currbase_ptr = seq + j;
            if (*currbase_ptr != bid) mmcnt++;
            if (mmcnt <= g_param->d) {
                td[i].matching_bases[td[i].num_matching_bases].bid = currbase_ptr + 1;
                td[i].matching_bases[td[i].num_matching_bases].wt = wt[j];
                td[i].matching_bases[td[i].num_matching_bases].mmcnt = mmcnt;
                td[i].num_matching_bases++;
            }
        }

        //output
        td[i].mmprofile = (int **) malloc(sizeof(int*) * ((size_t) (g_param->d+1)));
        for (k=0; k<=g_param->d; k++) {
            td[i].mmprofile[k] = (int *) malloc(sizeof(int) * ((size_t) last_index));
            for(j=0; j<last_index; j++) { td[i].mmprofile[k][j] = 0; }
        }
    }
}

static void kmertree_dfs_pthread_init_par16(const gkm_data *da, const int last_index, KmerTree *tree, kmertree_dfs_pthread_t *td)
{
    int i, j, k;
    const int d = g_param->d;

    //process the first TWO level and initialize thread input & output variables
    for (i=0; i<MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE; i++) {
        //input
        int bid1 = (i>>2) + 1;
        int bid2 = (i%4) + 1;
        td[i].tree = tree;
        td[i].start_depth = 2;
        td[i].start_node_index = MAX_ALPHABET_SIZE + i + 1;
        td[i].num_matching_bases = 0;
        td[i].last_seqid = last_index;

        uint8_t *seq = da->seq;
        uint8_t *wt = da->wt;
        for (j=0; j<da->seqlen - g_param->L + 1; j++) {
            int mmcnt = 0;
            uint8_t *base_ptr1 = seq + j;
            uint8_t *base_ptr2 = base_ptr1 + 1;
            if (*base_ptr1 != bid1) mmcnt++;
            if (*base_ptr2 != bid2) mmcnt++;
            if (mmcnt <= d) {
                td[i].matching_bases[td[i].num_matching_bases].bid = base_ptr2 + 1;
                td[i].matching_bases[td[i].num_matching_bases].wt = wt[j];
                td[i].matching_bases[td[i].num_matching_bases].mmcnt = mmcnt;
                td[i].num_matching_bases++;
            }
        }

        //output
        td[i].mmprofile = (int **) malloc(sizeof(int*) * ((size_t) (d+1)));
        for (k=0; k<=d; k++) {
            td[i].mmprofile[k] = (int *) malloc(sizeof(int) * ((size_t) last_index));
            for(j=0; j<last_index; j++) { td[i].mmprofile[k][j] = 0; }
        }
    }
}

static void *kmertree_dfs_pthread(void *ptr)
{
    kmertree_dfs_pthread_t *td = (kmertree_dfs_pthread_t *) ptr;

    kmertree_dfs(td->tree, td->last_seqid, td->start_depth, td->start_node_index, td->matching_bases, td->num_matching_bases, td->mmprofile);

    return 0;
}

static void kmertree_dfs_pthread_process(kmertree_dfs_pthread_t *td, const int nthreads, const int start, const int end, double *res)
{
    int i, j, k;
    pthread_t threads[MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE];
    int rc[MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE];
    const int d = g_param->d;

    //run threads. i=0 will be executed later in the main process
    for (i=1; i<nthreads; i++) {
        rc[i] = pthread_create(&threads[i], NULL, kmertree_dfs_pthread, (void *) &td[i]);
        if (rc[i]) {
            clog_error(CLOG(LOGGER_ID), "failed to create thread. pthread_create() returned %d", rc[i]);
        } else {
            clog_trace(CLOG(LOGGER_ID), "thread %d was created.", i);
        }
    }

    for (i=0; i<nthreads; i++) {
        if (i == 0) {
            kmertree_dfs_pthread(&td[i]);
        } else {
            if (rc[i] == 0) {
                //wait thread return
                pthread_join(threads[i], NULL);
            } else {
                //if failed to run thread, execute the function in the main process
                kmertree_dfs_pthread(&td[i]);
            }
        }

        for (j=start; j<end; j++) {
            for (k=0; k<=d; k++) {
                res[j-start] += (g_weights[k] * td[i].mmprofile[k][j]);
            }
        }

        //free
        for (k=0; k<=d; k++) {
            free(td[i].mmprofile[k]);
        }
        free(td[i].mmprofile);
    }

}

/**************************
 * kmertreecoef functions *
 **************************/
static void kmertreecoef_init(KmerTreeCoef *tree, int kmerlen)
{
    int i;
    tree->depth = kmerlen;
    tree->node_count = NODE_COUNT(kmerlen);
    tree->leaf_count = LEAF_COUNT(kmerlen);
    tree->coef_sum = (double *) malloc(sizeof(double) * ((size_t) tree->leaf_count));

    for (i=0; i<tree->leaf_count; i++) { tree->coef_sum[i] = 0; }
}

static void kmertreecoef_destroy(KmerTreeCoef *tree)
{
    if (tree) {
        free(tree->coef_sum);
        free(tree);
    }
}

static double kmertreecoef_dfs(const KmerTreeCoef *tree, const int depth, const int curr_node_index, const BaseMismatchCount *curr_matching_bases, const int curr_num_matching_bases)
{
    int j;
    int bid;
    double result = 0;
    const int d = g_param->d;

    if (depth == tree->depth - 1) {
        const double *coef= tree->coef_sum + (curr_node_index*MAX_ALPHABET_SIZE) - tree->node_count;
        for (bid=1; bid<=MAX_ALPHABET_SIZE; bid++) {
            for (j=0; j<curr_num_matching_bases; j++) {
                const uint8_t *currbase_ptr = curr_matching_bases[j].bid;
                const uint8_t currbase_wt = curr_matching_bases[j].wt;
                const int currbase_mmcnt = curr_matching_bases[j].mmcnt;
                if (*currbase_ptr == bid) {
                    // matching
                    result += (coef[bid] * g_weights[currbase_mmcnt] * currbase_wt);
                } else if (currbase_mmcnt < d) {
                    // non-matching
                    result += (coef[bid] * g_weights[currbase_mmcnt+1] * currbase_wt);
                }
            }
        }
        return result;
    } else {
        int daughter_node_index = (curr_node_index*MAX_ALPHABET_SIZE);
        for (bid=1; bid<=MAX_ALPHABET_SIZE; bid++) {
            daughter_node_index++;
            BaseMismatchCount next_matching_bases[MAX_SEQ_LENGTH];
            int next_num_matching_bases = 0;
            for (j=0; j<curr_num_matching_bases; j++) {
                uint8_t *currbase_ptr = curr_matching_bases[j].bid;
                int currbase_mmcnt = curr_matching_bases[j].mmcnt;
                if (*currbase_ptr == bid) {
                    // matching
                    next_matching_bases[next_num_matching_bases].bid = currbase_ptr+1;
                    next_matching_bases[next_num_matching_bases].wt = curr_matching_bases[j].wt;
                    next_matching_bases[next_num_matching_bases].mmcnt = currbase_mmcnt;
                    next_num_matching_bases++;
                } else if (currbase_mmcnt < d) {
                    // non-matching
                    next_matching_bases[next_num_matching_bases].bid = currbase_ptr+1;
                    next_matching_bases[next_num_matching_bases].wt = curr_matching_bases[j].wt;
                    next_matching_bases[next_num_matching_bases].mmcnt = currbase_mmcnt+1;
                    next_num_matching_bases++;
                }
            }

            if (next_num_matching_bases > 0) {
                result += kmertreecoef_dfs(tree, depth+1, daughter_node_index, next_matching_bases, next_num_matching_bases);
            } 
        }
        return result;
    }
}

static double kmertreecoef_dfs_single(const gkm_data *da)
{
    int i;
    BaseMismatchCount matching_bases[MAX_SEQ_LENGTH];
    int num_matching_bases = da->seqlen - g_param->L + 1;

    for (i=0; i<num_matching_bases; i++) {
        matching_bases[i].bid = da->seq + i;
        matching_bases[i].wt = da->wt[i];
        matching_bases[i].mmcnt = 0;
    }

    return kmertreecoef_dfs(g_sv_kmertreecoef, 0, 0, matching_bases, num_matching_bases);
}

static void kmertreecoef_dfs_pthread_init_par4(const gkm_data *da, kmertreecoef_dfs_pthread_t *td)
{
    int i, j;
    const int d= g_param->d;

    //process the first level and initialize thread input & output variables
    for (i=0; i<MAX_ALPHABET_SIZE; i++) {
        //input
        int bid = i + 1;
        td[i].start_depth = 1;
        td[i].start_node_index = bid;
        td[i].num_matching_bases = 0;

        uint8_t *seq = da->seq;
        uint8_t *wt = da->wt;
        for (j=0; j<da->seqlen - g_param->L + 1; j++) {
            int mmcnt = 0;
            uint8_t *currbase_ptr = seq + j;
            if (*currbase_ptr != bid) mmcnt++;
            if (mmcnt <= d) {
                td[i].matching_bases[td[i].num_matching_bases].bid = currbase_ptr + 1;
                td[i].matching_bases[td[i].num_matching_bases].wt = wt[j];
                td[i].matching_bases[td[i].num_matching_bases].mmcnt = mmcnt;
                td[i].num_matching_bases++;
            }
        }

        //output
        td[i].result = 0;
    }
}

static void kmertreecoef_dfs_pthread_init_par16(const gkm_data *da, kmertreecoef_dfs_pthread_t *td)
{
    int i, j;
    const int d = g_param->d;

    //process the first TWO level and initialize thread input & output variables
    for (i=0; i<MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE; i++) {
        //input
        int bid1 = (i>>2) + 1;
        int bid2 = (i%4) + 1;
        td[i].start_depth = 2;
        td[i].start_node_index = MAX_ALPHABET_SIZE + i + 1;
        td[i].num_matching_bases = 0;

        uint8_t *seq = da->seq;
        uint8_t *wt = da->wt;
        for (j=0; j<da->seqlen - g_param->L + 1; j++) {
            int mmcnt = 0;
            uint8_t *base_ptr1 = seq + j;
            uint8_t *base_ptr2 = base_ptr1 + 1;
            if (*base_ptr1 != bid1) mmcnt++;
            if (*base_ptr2 != bid2) mmcnt++;
            if (mmcnt <= d) {
                td[i].matching_bases[td[i].num_matching_bases].bid = base_ptr2 + 1;
                td[i].matching_bases[td[i].num_matching_bases].wt = wt[j];
                td[i].matching_bases[td[i].num_matching_bases].mmcnt = mmcnt;
                td[i].num_matching_bases++;
            }
        }

        //output
        td[i].result = 0;
    }
}

static void *kmertreecoef_dfs_pthread(void *ptr)
{
    kmertreecoef_dfs_pthread_t *td = (kmertreecoef_dfs_pthread_t *) ptr;

    td->result = kmertreecoef_dfs(g_sv_kmertreecoef, td->start_depth, td->start_node_index, td->matching_bases, td->num_matching_bases);

    return 0;
}

static double kmertreecoef_dfs_pthread_process(kmertreecoef_dfs_pthread_t *td, const int nthreads)
{
    int i;
    pthread_t threads[MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE];
    int rc[MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE];

    //run threads. i=0 will be executed later in the main process
    for (i=1; i<nthreads; i++) {
        rc[i] = pthread_create(&threads[i], NULL, kmertreecoef_dfs_pthread, (void *) &td[i]);
        if (rc[i]) {
            clog_error(CLOG(LOGGER_ID), "failed to create thread %d. pthread_create() returned %d", i, rc[i]);
        } else {
            clog_trace(CLOG(LOGGER_ID), "thread %d was created", i);
        }
    }

    //collect results
    double res = 0;
    for (i=0; i<nthreads; i++) {
        if (i == 0) {
            kmertreecoef_dfs_pthread(&td[i]);
        } else {
            if (rc[i] == 0) {
                //wait thread return
                pthread_join(threads[i], NULL);
            } else {
                //if failed to run thread, execute the function in the main process
                kmertreecoef_dfs_pthread(&td[i]);
            }
        }

        res += td[i].result;
    }

    return res;
}

static double kmertreecoef_dfs_par4(const gkm_data *da)
{
    kmertreecoef_dfs_pthread_t td[MAX_ALPHABET_SIZE];

    kmertreecoef_dfs_pthread_init_par4(da, td);

    return kmertreecoef_dfs_pthread_process(td, MAX_ALPHABET_SIZE);
}

static double kmertreecoef_dfs_par16(const gkm_data *da)
{
    kmertreecoef_dfs_pthread_t td[MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE];

    kmertreecoef_dfs_pthread_init_par16(da, td);

    return kmertreecoef_dfs_pthread_process(td, MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE);
}

/***************************************
 * gkmkernel internal kernel functions *
 ***************************************/
static void gkmkernel_kernelfunc_batch_single(const gkm_data *da, int n, double *res) 
{
    int i, j, k;
    BaseMismatchCount matching_bases[MAX_SEQ_LENGTH];
    int num_matching_bases = da->seqlen - g_param->L + 1;
    const int d = g_param->d;

    for (i=0; i<num_matching_bases; i++) {
        matching_bases[i].bid = da->seq + i;
        matching_bases[i].wt = da->wt[i];
        matching_bases[i].mmcnt = 0;
    }

    int **mmprofile = (int **) malloc(sizeof(int*) * ((size_t) (d+1)));
    for (k=0; k<=d; k++) {
        mmprofile[k] = (int *) malloc(sizeof(int)* ((size_t) n));
        for(j=0; j<n; j++) { mmprofile[k][j] = 0; }
    }

    kmertree_dfs(g_kmertree, n, 0, 0, matching_bases, num_matching_bases, mmprofile);

    for (j=0; j<n; j++) {
        res[j] = 0;
        for (k=0; k<=d; k++) {
            res[j] += (g_weights[k] * mmprofile[k][j]);
        }
    }

    for (k=0; k<=d; k++) {
        free(mmprofile[k]);
    }
    free(mmprofile);
}

static void gkmkernel_kernelfunc_batch_single_all(const gkm_data *da, KmerTree *tree, const int start, const int end, double *res) 
{
    int i, j, k;
    BaseMismatchCount matching_bases[MAX_SEQ_LENGTH];
    int num_matching_bases = da->seqlen - g_param->L + 1;
    const int d = g_param->d;

    for (i=0; i<num_matching_bases; i++) {
        matching_bases[i].bid = da->seq + i;
        matching_bases[i].wt = da->wt[i];
        matching_bases[i].mmcnt = 0;
    }

    /* initialize mmprofile*/
    int **mmprofile = (int **) malloc(sizeof(int*) * ((size_t) (g_param->d+1)));
    for (k=0; k<=g_param->d; k++) {
        mmprofile[k] = (int *) malloc(sizeof(int)* ((size_t) end));
        for(j=0; j<end; j++) { mmprofile[k][j] = 0; }
    }

    kmertree_dfs(tree, end, 0, 0, matching_bases, num_matching_bases, mmprofile);

    for (j=start; j<end; j++) {
        double sum = 0;
        for (k=0; k<=d; k++) {
            sum += (g_weights[k]*mmprofile[k][j]);
        }
        res[j-start] = sum;
    }

    //free mmprofile
    for (k=0; k<=d; k++) {
        free(mmprofile[k]);
    }
    free(mmprofile);
}

static void gkmkernel_kernelfunc_batch_par4(const gkm_data *da, int n, double *res) 
{
    kmertree_dfs_pthread_t td[MAX_ALPHABET_SIZE];

    kmertree_dfs_pthread_init_par4(da, n, g_kmertree, td);

    kmertree_dfs_pthread_process(td, MAX_ALPHABET_SIZE, 0, n, res);
}

static void gkmkernel_kernelfunc_batch_par16(const gkm_data *da, int n, double *res) 
{
    kmertree_dfs_pthread_t td[MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE];

    kmertree_dfs_pthread_init_par16(da, n, g_kmertree, td);

    kmertree_dfs_pthread_process(td, MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE, 0, n, res);
}

static void gkmkernel_kernelfunc_batch_par4_all(const gkm_data *da, KmerTree *tree, const int start, const int end, double *res)
{
    kmertree_dfs_pthread_t td[MAX_ALPHABET_SIZE];

    kmertree_dfs_pthread_init_par4(da, end, tree, td);

    kmertree_dfs_pthread_process(td, MAX_ALPHABET_SIZE, start, end, res);
}

static void gkmkernel_kernelfunc_batch_par16_all(const gkm_data *da, KmerTree *tree, int start, int end, double *res)
{
    kmertree_dfs_pthread_t td[MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE];

    kmertree_dfs_pthread_init_par16(da, end, tree, td);

    kmertree_dfs_pthread_process(td, MAX_ALPHABET_SIZE*MAX_ALPHABET_SIZE, start, end, res);
}

static double gkmkernel_kernelfunc_raw(const gkm_data *da, const gkm_data *db)
{
    double res = 0;

    kmertree_add_sequence(g_kmertree, 0, db);

    if (g_param_nthreads == 1) {
        gkmkernel_kernelfunc_batch_single(da, 1, &res);
    } else if (g_param_nthreads == 4) {
        gkmkernel_kernelfunc_batch_par4(da, 1, &res);
    } else if (g_param_nthreads == 16) {
        gkmkernel_kernelfunc_batch_par16(da, 1, &res);
    } else {
        clog_warn(CLOG(LOGGER_ID), "Supported number of threads are 1, 4 and 16. nthread is set to 1");
        g_param_nthreads = 1;
        gkmkernel_kernelfunc_batch_single(da, 1, &res);
    }

    kmertree_cleanup(g_kmertree, 0, 0);

    return res;
}

/******************************
 * global gkmkernel functions *
 ******************************/
/* build a new gkm_data structure */
gkm_data* gkmkernel_new_object(char *seq, char *sid, int seqid)
{
    gkm_data *d;
    int i, j, k;

    /* construct a feature vector */
    d = (gkm_data *) malloc(sizeof(gkm_data));

    if (sid) {
        d->sid = (char *) malloc(sizeof(char) * ((size_t) (strlen(sid) + 1)));
        strcpy(d->sid, sid);
    } else {
        d->sid = NULL;
    }
    d->seqid = seqid;
    d->seqlen = (int) strlen(seq);

    d->seq_string = (char *) malloc(sizeof(char) * ((size_t) (d->seqlen + 1)));
    strcpy(d->seq_string, seq);

    d->seq = (uint8_t *) malloc(sizeof(uint8_t) * ((size_t) d->seqlen));

    /* convert base to 0123 code */
    for (i=0; i<d->seqlen; i++) {
        switch (toupper(seq[i])) {
            case 'A': d->seq[i] = 1; break;
            case 'C': d->seq[i] = 2; break;
            case 'G': d->seq[i] = 3; break;
            case 'T': d->seq[i] = 4; break;
            default: 
                d->seq[i] = 1; 
                clog_warn(CLOG(LOGGER_ID), "'%c' at %s(%d) is not a valid nucleotide. Only ACGT are allowed", seq[i], sid, i);
                break;
        }
    }

    /* generate reverse complement sequence */
    d->seq_rc = (uint8_t *) malloc(sizeof(uint8_t) * ((size_t) d->seqlen));
    for (i=0; i<d->seqlen; i++) {
        //d->seq_rc[i] = (uint8_t) 5 - d->seq[d->seqlen-i-1]; 
        switch (d->seq[d->seqlen-i-1]) {
            case 1: d->seq_rc[i] = 4; break;
            case 2: d->seq_rc[i] = 3; break;
            case 3: d->seq_rc[i] = 2; break;
            case 4: d->seq_rc[i] = 1; break;
            default: d->seq_rc[i] = 1; break;
        }
    }

    /* convert sequence/sequence_rc to a set of k-mer ids */
    int nkmerids= (d->seqlen - g_param->L + 1);
    uint8_t *seqs[2] = {d->seq, d->seq_rc};
    d->kmerids = (int *) malloc(sizeof(int) * ((size_t) nkmerids));
    d->kmerids_rc = (int *) malloc(sizeof(int) * ((size_t) nkmerids));

    int *kmerids[2] = {d->kmerids, d->kmerids_rc};
    int total_node_count = NODE_COUNT(g_param->L);
    for (k=0; k<2; k++) {
        uint8_t *seq = seqs[k];
        int *kmerid = kmerids[k];
        for (j=0; j<nkmerids; j++) {
            int node_index = 0;
            for (i=0; i<g_param->L; i++) {
                node_index = (node_index*MAX_ALPHABET_SIZE) + seq[i+j];
            }
            kmerid[j] = node_index - total_node_count;
        }
    }

    d->wt = (uint8_t *) malloc(sizeof(uint8_t) * ((size_t) (d->seqlen - g_param->L + 1)));
    d->wt_rc = (uint8_t *) malloc(sizeof(uint8_t) * ((size_t) (d->seqlen - g_param->L + 1)));
    int center = nkmerids/2;

    if (g_param->kernel_type == EST_TRUNC_PW || g_param->kernel_type == EST_TRUNC_PW_RBF) {
        /* exponential decay weights */
        double H = g_param->H;
        uint8_t M = g_param->M;

        for (i=0; i<nkmerids; i++) { 
            uint8_t wt = (uint8_t) floor(M*exp((-1)*log(2)*abs(center-i)/H) + 1);
            if (wt>M) {wt=M;}
            d->wt[i] = wt;
            d->wt_rc[nkmerids-i-1] = wt;
        }
    } else {
        /* uniform weights */
        for (i=0; i<nkmerids; i++) { 
            d->wt[i] = 1;
            d->wt_rc[nkmerids-i-1] = 1;
        }
    }
    
    /* gaussian weights */
    /*
    double bw = 150;
    double gamma = log(0.5)/(bw*bw);
    double scale = 4;

    for (i=0; i<nkmerids; i++) { 
        uint8_t wt = (uint8_t) floor(scale*exp(gamma*(center-i)*(center-i)+0.02) + 1);
        d->wt[i] = wt;
        d->wt_rc[nkmerids-i-1] = wt;
    }
    */

    /* linear weights */
    /*
    double binsize = 50.0;
    for (i=0; i<nkmerids; i++) { 
        uint8_t wt = (uint8_t) floor((center - abs(center - i))/binsize + 1); 
        d->wt[i] = wt;
        d->wt_rc[nkmerids-i-1] = wt;
    }
    */

    /* calculate square root of the kernel(d,d) and store for normalization */
    double kern = gkmkernel_kernelfunc_raw(d, d);
	clog_trace(CLOG(LOGGER_ID), "%d's kernel is %f", seqid, kern);
    d->sqnorm = sqrt(kern);

    return(d);
}

/* free memory associated with the given object */
void gkmkernel_delete_object(gkm_data* d)
{
    free(d->kmerids);
    free(d->kmerids_rc);
    free(d->seq_string);
    free(d->wt);
    free(d->wt_rc);
    free(d->seq);
    free(d->seq_rc);
    if (d->sid) free(d->sid);
    free(d);
}

/* set the extra parameters for gkmkernel */
void gkmkernel_init(struct svm_parameter *param)
{
    int i;

    g_param = param;

    /* calculate the corresponding weights for calculating kernels from mismatch profiles
     *
     * 0: gkm-kernel
     * 1: gkm-kernel with estimated l-mers and non-truncated filter
     * 2: gkm-kernel with estimated l-mers and truncated filter (default)
     * 3: truncated filter + positional weights
     * 4: truncated filter + rbf
     * 5: truncated filter + positional weights + rbf
     */
    switch(g_param->kernel_type) {
        case GKM:
            calc_gkm_kernel_wt();
            break;
        case EST_FULL:
            calc_gkm_kernel_lmerest_wt(0);
            break;
        case EST_TRUNC:
            calc_gkm_kernel_lmerest_wt(1);
            break;
        case EST_TRUNC_PW:
            calc_gkm_kernel_lmerest_wt(1);
            break;
        case EST_TRUNC_RBF:
            calc_gkm_kernel_lmerest_wt(1);
            break;
        case EST_TRUNC_PW_RBF:
            calc_gkm_kernel_lmerest_wt(1);
            break;
        default:
            calc_gkm_kernel_lmerest_wt(1);
            break;
    }

    clog_debug(CLOG(LOGGER_ID), "gkm-kernel weights:");

    for (i=0; i<=g_param->d; i++) {
        clog_debug(CLOG(LOGGER_ID), "  c[%d] = %.6f", i, g_weights[i]);     
    }

    g_kmertree = (KmerTree *) malloc(sizeof(KmerTree));
    kmertree_init(g_kmertree, g_param->L);
}

void gkmkernel_init_problems(union svm_data *x, int n)
{
    int i;

    /* initialize g_prob_kmertree */
    g_prob_kmertree = (KmerTree *) malloc(sizeof(KmerTree));
    kmertree_init(g_prob_kmertree, g_param->L);

    g_prob_svm_data = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) n));
    memcpy((void *)g_prob_svm_data, (void *)x, sizeof(union svm_data) * ((size_t) n));
    g_prob_num = n;

    g_prob_gkmkernel_index = (int *) malloc(sizeof(int) * ((size_t) n));
    g_prob_libsvm_index = (int *) malloc(sizeof(int) * ((size_t) n));

    //add sequences
    for (i=0; i<n; i++) {
        g_prob_gkmkernel_index[i] = i;
        g_prob_libsvm_index[i] = i;
        kmertree_add_sequence(g_prob_kmertree, i, x[i].d);
    }
}

void gkmkernel_init_sv(union svm_data *sv, double *coef, int nclass, int n) 
{
    int i, j, k;

    g_sv_svm_data = sv;
    g_sv_num = n;

    /* initialize g_sv_kmertree */
    g_sv_kmertree = (KmerTree *) malloc(sizeof(KmerTree));
    kmertree_init(g_sv_kmertree, g_param->L);

    //add SV sequences
    for (j=0; j<n; j++) {
        kmertree_add_sequence(g_sv_kmertree, j, sv[j].d);
    }

    g_sv_mmprofile = (int **) malloc(sizeof(int*) * ((size_t) (g_param->d+1)));
    for (i=0; i<=g_param->d; i++) {
        g_sv_mmprofile[i] = (int *) malloc(sizeof(int) * ((size_t) n));
        for(j=0; j<n; j++) { g_sv_mmprofile[i][j] = 0; }
    }

    //speed-up for linear cases
    if ((nclass == 2) && (g_param->kernel_type != EST_TRUNC_RBF) && (g_param->kernel_type != EST_TRUNC_PW_RBF)) {
        g_sv_kmertreecoef = (KmerTreeCoef *) malloc(sizeof(KmerTreeCoef));
        kmertreecoef_init(g_sv_kmertreecoef, g_param->L);

        //add (normalized) sv coef to the corresponding leaf
        for (j=0; j<n; j++) {
            gkm_data *sv_j = sv[j].d;
            double sv_coef = coef[j];
            int nkmerids = sv_j->seqlen - g_param->L + 1;

            int *kmerids[2] = {sv_j->kmerids, sv_j->kmerids_rc};
            uint8_t *wts[2] = {sv_j->wt, sv_j->wt_rc};
            for (k=0; k<2; k++) {
                int *kmerid = kmerids[k];
                uint8_t *wt = wts[k];
                for (i=0; i<nkmerids; i++) {
                    g_sv_kmertreecoef->coef_sum[kmerid[i]] += ((sv_coef*wt[i])/sv_j->sqnorm);
                }
            }
        }
    }
}

void gkmkernel_destroy_sv()
{
    kmertree_destroy(g_sv_kmertree);
    kmertreecoef_destroy(g_sv_kmertreecoef);

    if (g_sv_mmprofile) {
        int i;
        for (i=0; i<=g_param->d; i++) {
            free(g_sv_mmprofile[i]);
        }
        free(g_sv_mmprofile);
    }

    g_sv_kmertree = NULL;
    g_sv_kmertreecoef = NULL;
    g_sv_mmprofile = NULL;
    g_sv_svm_data = NULL;
    g_sv_num = 0;
}

void gkmkernel_destroy_problems()
{
    kmertree_destroy(g_prob_kmertree);

    if (g_prob_svm_data) { free(g_prob_svm_data); }
    if (g_prob_gkmkernel_index) { free(g_prob_gkmkernel_index); }
    if (g_prob_libsvm_index) { free(g_prob_libsvm_index); }

    g_prob_kmertree = NULL;
    g_prob_svm_data = NULL;
    g_prob_gkmkernel_index = NULL;
    g_prob_libsvm_index = NULL;
    g_prob_num = 0;
}

void gkmkernel_destroy()
{
    gkmkernel_destroy_problems();
    gkmkernel_destroy_sv();

    kmertree_destroy(g_kmertree);

    g_kmertree = NULL;
}


void gkmkernel_swap_index(int i, int j)
{
    int tmp;

    tmp                                            = g_prob_libsvm_index[g_prob_gkmkernel_index[i]];
    g_prob_libsvm_index[g_prob_gkmkernel_index[i]] = g_prob_libsvm_index[g_prob_gkmkernel_index[j]];
    g_prob_libsvm_index[g_prob_gkmkernel_index[j]] = tmp;

    tmp                       = g_prob_gkmkernel_index[i];
    g_prob_gkmkernel_index[i] = g_prob_gkmkernel_index[j];
    g_prob_gkmkernel_index[j] = tmp;
}

void gkmkernel_update_index()
{
    int i, j;
    for (i=0; i<g_prob_kmertree->leaf_count; i++) {
        KmerTreeLeaf *leaf = g_prob_kmertree->leaf + i;
        KmerTreeLeafData *data = leaf->data;
        for (j=0; j<leaf->count; j++) {
            int old_id = data[j].seqid;
            data[j].seqid = g_prob_libsvm_index[old_id];
        }
    }

    union svm_data *svm_data_new = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) g_prob_num));
    for (i=0; i<g_prob_num; i++) {
        svm_data_new[i] = g_prob_svm_data[g_prob_gkmkernel_index[i]];
    }

    free(g_prob_svm_data);
    g_prob_svm_data = svm_data_new;

    //reset
    for (i=0; i<g_prob_num; i++) {
        g_prob_gkmkernel_index[i] = i;
        g_prob_libsvm_index[i] = i;
    }
}



/********************
 * kernel functions *
 ********************/
/* single kernel */
double gkmkernel_kernelfunc(const gkm_data *da, const gkm_data *db)
{
    if (da == db) {
        return 1.0;
    } else {
        return gkmkernel_kernelfunc_raw(da, db)/(da->sqnorm*db->sqnorm);
    }
}

/* calculate multiple kernels when n is relatively small */
double* gkmkernel_kernelfunc_batch(const gkm_data *da, const union svm_data *db_array, const int n, double *res) 
{
    int i, j;
    struct timeval time_start, time_end;

    //add sequences to the tree
    gettimeofday(&time_start, NULL);
    for (i=0; i<n; i++) {
        kmertree_add_sequence(g_kmertree, i, db_array[i].d);
    }
    gettimeofday(&time_end, NULL);
    clog_debug(CLOG(LOGGER_ID), "add sequences to kmertree (%ld ms)", diff_ms(time_end, time_start));

    gettimeofday(&time_start, NULL);
    //initialize result variable
    for (j=0; j<n; j++) { res[j] = 0; }

    if (g_param_nthreads == 1) {
        gkmkernel_kernelfunc_batch_single(da, n, res);
    } else if (g_param_nthreads == 4) {
        gkmkernel_kernelfunc_batch_par4(da, n, res);
    } else if (g_param_nthreads == 16) {
        gkmkernel_kernelfunc_batch_par16(da, n, res);
    } else {
        clog_warn(CLOG(LOGGER_ID), "Supported number of threads are 1, 4 and 16. nthread is set to 1");
        g_param_nthreads = 1;
        gkmkernel_kernelfunc_batch_single(da, n, res);
    }

    //normalization
    double da_sqnorm = da->sqnorm;
    for (i=0; i<n; i++) {
        res[i] /= (da_sqnorm*db_array[i].d->sqnorm);
    }

    //RBF kernel
    if (g_param->kernel_type == EST_TRUNC_RBF || g_param->kernel_type == EST_TRUNC_PW_RBF) {
        for (i=0; i<n; i++) {
            res[i] = exp(g_param->gamma*(res[i]-1));
        }
    }

    gettimeofday(&time_end, NULL);
    clog_trace(CLOG(LOGGER_ID), "DFS n=%d (%ld ms)", n, diff_ms(time_end, time_start));

    kmertree_cleanup(g_kmertree, 0, 0);

    return res;
}

/* calculate multiple kernels using precomputed kmertree with all samples */
double* gkmkernel_kernelfunc_batch_all(const int a, const int start, const int end, double *res) 
{
    int j;
    const gkm_data *da = g_prob_svm_data[a].d;
    struct timeval time_start, time_end;

    gettimeofday(&time_start, NULL);

    //initialize result variable
    for (j=0; j<end-start; j++) { res[j] = 0; }

    if (g_param_nthreads == 1) {
        gkmkernel_kernelfunc_batch_single_all(da, g_prob_kmertree, start, end, res);
    } else if (g_param_nthreads == 4) {
        gkmkernel_kernelfunc_batch_par4_all(da, g_prob_kmertree, start, end, res);
    } else if (g_param_nthreads == 16) {
        gkmkernel_kernelfunc_batch_par16_all(da, g_prob_kmertree, start, end, res);
    }
    else {
        clog_warn(CLOG(LOGGER_ID), "Supported number of threads are 1, 4 and 16. nthread is set to 1");
        g_param_nthreads = 1;
        gkmkernel_kernelfunc_batch_single_all(da, g_prob_kmertree, start, end, res);
    }

    //normalization
    double da_sqnorm = da->sqnorm;
    for (j=start; j<end; j++) {
        res[j-start] /= (da_sqnorm*g_prob_svm_data[j].d->sqnorm);
    }

    //RBF kernel
    if (g_param->kernel_type == EST_TRUNC_RBF || g_param->kernel_type == EST_TRUNC_PW_RBF) {
        for (j=0; j<end-start; j++) {
            res[j] = exp(g_param->gamma*(res[j]-1));
        }
    }

    gettimeofday(&time_end, NULL);
    clog_trace(CLOG(LOGGER_ID), "DFS i=%d, start=%d, end=%d (%ld ms)", a, start, end, diff_ms(time_end, time_start));

    return res;
}

/* calculate multiple kernels using precomputed kmertree with SVs */
double* gkmkernel_kernelfunc_batch_sv(const gkm_data *da, double *res) 
{
    if (g_sv_kmertree == NULL) {
        clog_error(CLOG(LOGGER_ID), "kmertree for SVs has not been initialized. call gkmkernel_init_sv() first.");
        return NULL;
    }

    int i, j;
    struct timeval time_start, time_end;
    const int d = g_param->d;

    gettimeofday(&time_start, NULL);

    //initialize results
    for (j=0; j<g_sv_num; j++) { res[j] = 0; }

    if (g_param_nthreads == 1) {
        gkmkernel_kernelfunc_batch_single_all(da, g_sv_kmertree, 0, g_sv_num, res);
    } else if (g_param_nthreads == 4) {
        gkmkernel_kernelfunc_batch_par4_all(da, g_sv_kmertree, 0, g_sv_num, res);
    } else if (g_param_nthreads == 16) {
        gkmkernel_kernelfunc_batch_par16_all(da, g_sv_kmertree, 0, g_sv_num, res);
    }
    else {
        clog_warn(CLOG(LOGGER_ID), "Supported number of threads are 1, 4 and 16. nthread is set to 1");
        g_param_nthreads = 1;
        gkmkernel_kernelfunc_batch_single_all(da, g_sv_kmertree, 0, g_sv_num, res);
    }

    //normalization
    double da_sqnorm = da->sqnorm;
    for (j=0; j<g_sv_num; j++) {
        res[j] /= (da_sqnorm*g_sv_svm_data[j].d->sqnorm);
    }

    //RBF kernel
    if (g_param->kernel_type == EST_TRUNC_RBF || g_param->kernel_type == EST_TRUNC_PW_RBF) {
        for (j=0; j<g_sv_num; j++) {
            res[j] = exp(g_param->gamma*(res[j]-1));
        }
    }

    //reset mismatch profiles
    for (i=0; i<=d; i++) {
        for(j=0; j<g_sv_num; j++) { g_sv_mmprofile[i][j] = 0; }
    }

    gettimeofday(&time_end, NULL);
    clog_trace(CLOG(LOGGER_ID), "DFS nSVs=%d (%ld ms)", g_sv_num, diff_ms(time_end, time_start));

    return res;
}

double gkmkernel_predict(const gkm_data *d)
{
    double result = 0;

    if (g_sv_kmertreecoef == NULL) {
        clog_error(CLOG(LOGGER_ID), "kmertreecoef has not been initialized. call gkmkernel_init_sv() first.\n");
        return 0;
    }

    if (g_param_nthreads == 1) {
        result = kmertreecoef_dfs_single(d);
    } else if (g_param_nthreads == 4) {
        result = kmertreecoef_dfs_par4(d);
    } else if (g_param_nthreads == 16) {
        result = kmertreecoef_dfs_par16(d);
    }
    else {
        clog_warn(CLOG(LOGGER_ID), "Supported number of threads are 1, 4 and 16. nthread is set to 1");
        g_param_nthreads = 1;
        result = kmertreecoef_dfs_single(d);
    }

    return result/d->sqnorm;
}

void gkmkernel_set_num_threads(int n)
{
    if ((n == 1) || (n == 4) || (n == 16)) {
        clog_info(CLOG(LOGGER_ID), "Number of threads is set to %d", n);
        g_param_nthreads = n;
    } else {
        clog_warn(CLOG(LOGGER_ID), "Supported number of threads are 1, 4 and 16. nthread is set to 1");
        g_param_nthreads = 1;
    }
}
