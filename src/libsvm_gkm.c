/* libsvm_gkm.c
 *
 * Copyright (C) 2020 Dongwon Lee
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
#include <locale.h>

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
static double g_weights[MAX_MM+1] = {0.0};

static KmerTree *g_kmertree = NULL;

static KmerTree *g_prob_kmertree = NULL;
static union svm_data *g_prob_svm_data = NULL;
static int g_prob_num = 0;
static int *g_prob_gkmkernel_index = NULL;
static int *g_prob_libsvm_index = NULL;

static KmerTree *g_sv_kmertree = NULL;
static union svm_data *g_sv_svm_data = NULL;
static int g_sv_num = 0;

static KmerTreeCoef *g_sv_kmertreecoef = NULL;

static uint8_t *g_mmcnt_lookuptab = NULL;
static int g_mmcnt_lookuptab_mask = 0;
static int g_mmcnt_nlookups = 0;

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

typedef struct _kernelfunc_sqnorm_pthread_t {
    int *twobitids;
    int *wt;
    int *mmprofile;
    int nids;
    int start_idx;
    int end_idx;
} kernelfunc_sqnorm_pthread_t;

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
    int RC = (g_param->norc==1) ? 1 : 2;

    for (k=0; k<RC; k++) {
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
    pthread_t threads[MAX_ALPHABET_SIZE_SQ];
    int rc[MAX_ALPHABET_SIZE_SQ];
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

        //speed-up (~20%) by rearranging the double-loop and pre-calculation
        for (j=0; j<curr_num_matching_bases; j++) {
            const BaseMismatchCount *currbase = curr_matching_bases+j;
            const uint8_t currbase_bid = *currbase->bid;
            const int currbase_mmcnt = currbase->mmcnt;
            const double currbase_mmcnt_wt0 = g_weights[currbase_mmcnt] * currbase->wt;
            const double currbase_mmcnt_wt1 = g_weights[currbase_mmcnt+1] * currbase->wt;
            for (bid=1; bid<=MAX_ALPHABET_SIZE; bid++) {
                if (currbase_bid == bid) {
                    // matching
                    result += (coef[bid] * currbase_mmcnt_wt0);
                } else if (currbase_mmcnt < d) {
                    // non-matching
                    result += (coef[bid] * currbase_mmcnt_wt1);
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
    pthread_t threads[MAX_ALPHABET_SIZE_SQ];
    int rc[MAX_ALPHABET_SIZE_SQ];

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
    kmertreecoef_dfs_pthread_t td[MAX_ALPHABET_SIZE_SQ];

    kmertreecoef_dfs_pthread_init_par16(da, td);

    return kmertreecoef_dfs_pthread_process(td, MAX_ALPHABET_SIZE_SQ);
}

/***************************************
 * gkmkernel internal kernel functions *
 ***************************************/
static void gkmkernel_kernelfunc_batch_single(const gkm_data *da, KmerTree *tree, const int start, const int end, double *res) 
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
    int **mmprofile = (int **) malloc(sizeof(int*) * ((size_t) (d+1)));
    for (k=0; k<=d; k++) {
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

static void gkmkernel_kernelfunc_batch_par4(const gkm_data *da, KmerTree *tree, const int start, const int end, double *res)
{
    kmertree_dfs_pthread_t td[MAX_ALPHABET_SIZE];

    kmertree_dfs_pthread_init_par4(da, end, tree, td);

    kmertree_dfs_pthread_process(td, MAX_ALPHABET_SIZE, start, end, res);
}

static void gkmkernel_kernelfunc_batch_par16(const gkm_data *da, KmerTree *tree, int start, int end, double *res)
{
    kmertree_dfs_pthread_t td[MAX_ALPHABET_SIZE_SQ];

    kmertree_dfs_pthread_init_par16(da, end, tree, td);

    kmertree_dfs_pthread_process(td, MAX_ALPHABET_SIZE_SQ, start, end, res);
}

//function pointer for the three batch kernel functions
// gkmkernel_kernelfunc_batch_single
// gkmkernel_kernelfunc_batch_par4
// gkmkernel_kernelfunc_batch_par16
static void (*gkmkernel_kernelfunc_batch_ptr)(const gkm_data *da, KmerTree *tree, int start, int end, double *res) = gkmkernel_kernelfunc_batch_single;

static double gkmkernel_kernelfunc_raw(const gkm_data *da, const gkm_data *db)
{
    double res = 0;

    kmertree_add_sequence(g_kmertree, 0, db);

    gkmkernel_kernelfunc_batch_ptr(da, g_kmertree, 0, 1, &res);

    kmertree_cleanup(g_kmertree, 0, 0);

    return res;
}

// 2/17/2016
// functions for efficient calulation of sqrt(K(a, a)) using XOR lookup table
// This was also implemented in the original gkm-SVM software
// current implementation only supports L<=MMCNT_LOOKUPTAB_WIDTH*2
static void gkmkernel_build_mmcnt_lookuptable()
{
    int i, j;
    int mask = 3;
    int tablesize = (1<<(MMCNT_LOOKUPTAB_WIDTH*2));

    g_mmcnt_lookuptab = (uint8_t *) malloc(sizeof(uint8_t) * tablesize);

    for (i=0; i<tablesize; i++) {
        int xor_word = i;
        g_mmcnt_lookuptab[i] = 0;
        for (j=0; j<MMCNT_LOOKUPTAB_WIDTH; j++) {
            if ((xor_word & mask) != 0) {
                g_mmcnt_lookuptab[i]++;
            }
            xor_word >>= 2;
        }
        //clog_trace(CLOG(LOGGER_ID), "g_lookup_table: %d %d", i, g_mmcnt_lookuptab[i]);
    }

    g_mmcnt_lookuptab_mask = 0;
    for (i=0; i<MMCNT_LOOKUPTAB_WIDTH; i++) {
        g_mmcnt_lookuptab_mask = ((g_mmcnt_lookuptab_mask << 2) | 3);
    }
   
    if (g_param->L <= MMCNT_LOOKUPTAB_WIDTH) {
        g_mmcnt_nlookups = 1;
    } else if (g_param->L <= MMCNT_LOOKUPTAB_WIDTH*2) {
        g_mmcnt_nlookups = 2;
    } else {
        clog_error(CLOG(LOGGER_ID), "L(%d) cannot be greater than MMCNT_LOOKUPTAB_WIDTH*2 (%d).", g_param->L, MMCNT_LOOKUPTAB_WIDTH*2);
    }

    clog_trace(CLOG(LOGGER_ID), "g_mmcnt_lookuptab_mask: %x", g_mmcnt_lookuptab_mask);
    clog_trace(CLOG(LOGGER_ID), "g_mmcnt_nlookups: %d", g_mmcnt_nlookups);
}

static int sequence2twobitids(const gkm_data *d, int *twobitids)
{
    int i, j;
    uint8_t *seqs[2] = {d->seq, d->seq_rc};
    int RC = (g_param->norc==1) ? 1 : 2;
    int nids = (d->seqlen - g_param->L + 1) * RC;

    int mask=3;
    for (i=0; i<g_param->L-1; i++) {
        mask = ((mask<<2) | 3);
    }

    int cnt = 0;
    for (j=0; j<RC; j++) { 
        int twobitid=0;
        uint8_t *s = seqs[j];
        for (i=0; i<g_param->L-1; i++) {
            twobitid = ((twobitid<<2) | (s[i]-1));
            //clog_trace(CLOG(LOGGER_ID), "seq2twobitid: %d %d %x", i, s[i]-1, twobitid);
        }

        for (i=g_param->L-1; i<d->seqlen; i++) {
            twobitid = (((twobitid<<2) & mask) | (s[i]-1));

            twobitids[cnt] = (twobitid & g_mmcnt_lookuptab_mask);

            //each id is divided into two pieces if L is greater than MMCNT_LOOKUPTAB_WIDTH
            if (g_mmcnt_nlookups == 2) {
                twobitids[nids + cnt] = ((twobitid >> (2*MMCNT_LOOKUPTAB_WIDTH)) & g_mmcnt_lookuptab_mask);
            }

            cnt++;
            //clog_trace(CLOG(LOGGER_ID), "seq2twobitid: %d %d %x", i, s[i]-1, twobitid);
        }
    }

    return cnt;
}

static void *kernelfunc_sqnorm_pthread(void *ptr)
{
    int i, j;

    kernelfunc_sqnorm_pthread_t *td = (kernelfunc_sqnorm_pthread_t *) ptr;
    const int *twobitids = td->twobitids;
    const int *wt = td->wt;
    const int nids = td->nids;
    const int d = g_param->d;
    int *mmprofile = td->mmprofile;

    for (i=td->start_idx; i<td->end_idx; i++) {
        const int id0 = twobitids[i];
        const int id1 = twobitids[nids + i];
        const int wt_i = wt[i];
        for (j=0; j<nids; j++) {
            int mmcnt = g_mmcnt_lookuptab[id0 ^ twobitids[j]];
            if ((mmcnt <= d) && (g_mmcnt_nlookups == 2)) {
                mmcnt += g_mmcnt_lookuptab[id1 ^ twobitids[nids + j]];
            }
            if (mmcnt <= d) {
                mmprofile[mmcnt]+=(wt_i*wt[j]);
            }
        }
    }

    return 0;
}

static double gkmkernel_kernelfunc_sqnorm_single(const gkm_data *da)
{
    int i, j, k;
    int twobitids[MAX_SEQ_LENGTH*2*2];
    int wt[MAX_SEQ_LENGTH*2];
    int nids = sequence2twobitids(da, twobitids); 
    int nkmerids= (da->seqlen - g_param->L + 1);
    int d = g_param->d;

    int mmprofile[MAX_MM];

    for (k=0; k<=d; k++) { mmprofile[k] = 0; }
    for (i=0; i<nkmerids; i++) { wt[i] = da->wt[i]; }
    for (i=0; i<nkmerids; i++) { wt[nkmerids + i] = da->wt_rc[i]; }

    for (i=0; i<nkmerids; i++) { //the forward stand only
        const int id0 = twobitids[i];
        const int id1 = twobitids[nids + i];
        const int wt_i = wt[i];
        for (j=0; j<nids; j++) {
            int mmcnt = g_mmcnt_lookuptab[id0 ^ twobitids[j]];
            if ((mmcnt <= d) && (g_mmcnt_nlookups == 2)) {
                mmcnt += g_mmcnt_lookuptab[id1 ^ twobitids[nids + j]];
            }
            if (mmcnt <= d) {
                mmprofile[mmcnt]+=(wt_i*wt[j]);
            }
        }
    }

    double sum = 0;
    for (k=0; k<=d; k++) {
        sum += (g_weights[k]*mmprofile[k]);
    }

    return sqrt(sum);
}

static double gkmkernel_kernelfunc_sqnorm_multi(const gkm_data *da)
{
    int i, k;
    int twobitids[MAX_SEQ_LENGTH*2*2];
    int wt[MAX_SEQ_LENGTH*2];
    int nids = sequence2twobitids(da, twobitids);
    int nkmerids= (da->seqlen - g_param->L + 1);
    int d = g_param->d;

    kernelfunc_sqnorm_pthread_t td[MAX_ALPHABET_SIZE_SQ];
    pthread_t threads[MAX_ALPHABET_SIZE_SQ];
    int rc[MAX_ALPHABET_SIZE_SQ];
    int mmprofile[MAX_ALPHABET_SIZE_SQ][MAX_MM];

    for (i=0; i<g_param_nthreads; i++) {
        for (k=0; k<=d; k++) { mmprofile[i][k] = 0; }
    }
    for (i=0; i<nkmerids; i++) { wt[i] = da->wt[i]; }
    for (i=0; i<nkmerids; i++) { wt[nkmerids + i] = da->wt_rc[i]; }

    int prev_end_idx = 0;
    for (i=0; i<g_param_nthreads; i++) {
        td[i].twobitids = twobitids;
        td[i].wt = wt;
        td[i].mmprofile = mmprofile[i];
        td[i].nids = nids;
        td[i].start_idx = prev_end_idx;
        td[i].end_idx = int(nkmerids*(i+1)/g_param_nthreads) + 1;
        prev_end_idx = td[i].end_idx;
    }
    td[g_param_nthreads-1].end_idx = nkmerids; // last idx in the last thread should always be this.

    //run threads. i=0 will be executed later in the main process
    for (i=1; i<g_param_nthreads; i++) {
        rc[i] = pthread_create(&threads[i], NULL, kernelfunc_sqnorm_pthread, (void *) &td[i]);
        if (rc[i]) {
            clog_error(CLOG(LOGGER_ID), "failed to create thread %d. pthread_create() returned %d", i, rc[i]);
        } else {
            clog_trace(CLOG(LOGGER_ID), "thread %d was created", i);
        }
    }

    //collect results
    for (i=0; i<g_param_nthreads; i++) {
        if (i == 0) {
            kernelfunc_sqnorm_pthread((void *) &td[i]);
        } else {
            if (rc[i] == 0) {
                //wait thread return
                pthread_join(threads[i], NULL);
            } else {
                //if failed to run thread, execute the function in the main process
                kernelfunc_sqnorm_pthread((void *) &td[i]);
            }
        }
    }

    double sum = 0;
    for (k=0; k<=d; k++) {
        for (i=1; i<g_param_nthreads; i++) { mmprofile[0][k] += mmprofile[i][k]; }
        sum += (g_weights[k]*mmprofile[0][k]);
    }

    return sqrt(sum);
}

static double gkmkernel_kernelfunc_sqnorm(const gkm_data *da)
{
    if (g_param_nthreads == 1) {
        return gkmkernel_kernelfunc_sqnorm_single(da);
    } else {
        return gkmkernel_kernelfunc_sqnorm_multi(da);
    }
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
    
    /* calculate square root of the kernel(d,d) and store for normalization */
    d->sqnorm = gkmkernel_kernelfunc_sqnorm(d);
	clog_trace(CLOG(LOGGER_ID), "%d's sqnorm is %f", seqid, d->sqnorm);

    return(d);
}

/* free memory associated with the given object including the gkm_data object itself */
void gkmkernel_delete_object(gkm_data* d)
{
    if (d->kmerids) free(d->kmerids);
    if (d->kmerids_rc) free(d->kmerids_rc);
    if (d->seq_string) free(d->seq_string);
    if (d->wt) free(d->wt);
    if (d->wt_rc) free(d->wt_rc);
    if (d->seq) free(d->seq);
    if (d->seq_rc) free(d->seq_rc);
    if (d->sid) free(d->sid);

    free(d);
}

/* free up memory associated with the given object except the gkm_data to reduce memory usage in gkmpredict */
void gkmkernel_free_object(gkm_data* d)
{
    if (d->kmerids) free(d->kmerids);
    if (d->kmerids_rc) free(d->kmerids_rc);
    if (d->seq_string) free(d->seq_string);
    if (d->wt) free(d->wt);
    if (d->wt_rc) free(d->wt_rc);
    if (d->seq) free(d->seq);
    if (d->seq_rc) free(d->seq_rc);
    if (d->sid) free(d->sid);

    d->kmerids = NULL;
    d->kmerids_rc = NULL;
    d->seq_string = NULL;
    d->wt = NULL;
    d->wt_rc = NULL;
    d->seq = NULL;
    d->seq_rc = NULL;
    d->sid = NULL;
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

    gkmkernel_build_mmcnt_lookuptable();
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

static void gkmkernel_add_one_sv(gkm_data *sv_i, double sv_coef, int i, int nclass)
{
    int j, k;
    int RC = (g_param->norc==1) ? 1 : 2;

    if ((nclass == 2) &&
        (g_param->kernel_type != EST_TRUNC_RBF) &&
        (g_param->kernel_type != EST_TRUNC_PW_RBF)) {

        //add (normalized) sv coef to the corresponding leaf
        int nkmerids = sv_i->seqlen - g_param->L + 1;

        int *kmerids[2] = {sv_i->kmerids, sv_i->kmerids_rc};
        uint8_t *wts[2] = {sv_i->wt, sv_i->wt_rc};
        for (k=0; k<RC; k++) {
            int *kmerid = kmerids[k];
            uint8_t *wt = wts[k];
            for (j=0; j<nkmerids; j++) {
                g_sv_kmertreecoef->coef_sum[kmerid[j]] += ((sv_coef*wt[j])/sv_i->sqnorm);
            }
        }
    } else {
        kmertree_add_sequence(g_sv_kmertree, i, sv_i);
    }
}

static void gkmkernel_init_sv_kmertree_objects(int nclass)
{
    if ((nclass == 2) &&
        (g_param->kernel_type != EST_TRUNC_RBF) &&
        (g_param->kernel_type != EST_TRUNC_PW_RBF)) {
        //speed-up for linear binary classifier with g_sv_kemrtreecoef
        g_sv_kmertreecoef = (KmerTreeCoef *) malloc(sizeof(KmerTreeCoef));
        kmertreecoef_init(g_sv_kmertreecoef, g_param->L);
    } else {
        // initialize g_sv_kmertree for non-linear cases
        g_sv_kmertree = (KmerTree *) malloc(sizeof(KmerTree));
        kmertree_init(g_sv_kmertree, g_param->L);
    }
}

void gkmkernel_init_sv(union svm_data *sv, double *coef, int nclass, int n) 
{
    int i;

    gkmkernel_init_sv_kmertree_objects(nclass);

    for (i=0; i<n; i++) {
        gkmkernel_add_one_sv(sv[i].d, coef[i], i, nclass);
    }

    g_sv_svm_data = sv;
    g_sv_num = n;
}

void gkmkernel_destroy_sv()
{
    kmertree_destroy(g_sv_kmertree);
    kmertreecoef_destroy(g_sv_kmertreecoef);

    g_sv_kmertree = NULL;
    g_sv_kmertreecoef = NULL;
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

    gkmkernel_kernelfunc_batch_ptr(da, g_kmertree, 0, n, res);

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

    gkmkernel_kernelfunc_batch_ptr(da, g_prob_kmertree, start, end, res);

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

    int j;
    struct timeval time_start, time_end;

    gettimeofday(&time_start, NULL);

    //initialize results
    for (j=0; j<g_sv_num; j++) { res[j] = 0; }

    gkmkernel_kernelfunc_batch_ptr(da, g_sv_kmertree, 0, g_sv_num, res);

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
    g_param_nthreads = n;

    //clog_info(CLOG(LOGGER_ID), "Number of threads is set to %d", n);

    if (g_param_nthreads == 1) {
        gkmkernel_kernelfunc_batch_ptr = gkmkernel_kernelfunc_batch_single;
    } else if (g_param_nthreads == 4) {
        gkmkernel_kernelfunc_batch_ptr = gkmkernel_kernelfunc_batch_par4;
    } else if (g_param_nthreads == 16) {
        gkmkernel_kernelfunc_batch_ptr = gkmkernel_kernelfunc_batch_par16;
    } else {
        clog_warn(CLOG(LOGGER_ID), "Supported number of threads are 1, 4 and 16. nthread is set to 1");
        g_param_nthreads = 1;
        gkmkernel_kernelfunc_batch_ptr = gkmkernel_kernelfunc_batch_single;
    }
}

/*
 * variables and functions copied from libsvm.cpp to improve svm_load_model function
 *
 */
static const char *svm_type_table[] =
{
    "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
    "gkm_cnt", "gkm_estfull", "gkm_esttrunc", "gkmrbf", "wgkm", "wgkmrbf", NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
    FILE *fp = fopen(model_file_name,"w");
    if(fp==NULL) return -1;

    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    const svm_parameter& param = model->param;

    fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
    fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);
    fprintf(fp,"L %d\n", param.L);
    fprintf(fp,"k %d\n", param.k);
    fprintf(fp,"d %d\n", param.d);
    fprintf(fp,"norc %d\n", param.norc);

    if ((param.kernel_type == EST_TRUNC_RBF) || (param.kernel_type == EST_TRUNC_PW_RBF)) {
        fprintf(fp,"gamma %g\n", param.gamma);
    }

    if ((param.kernel_type == EST_TRUNC_PW) || (param.kernel_type == EST_TRUNC_PW_RBF)) {
        fprintf(fp,"M %d\n", param.M);
        fprintf(fp,"H %g\n", param.H);
    }

    int nr_class = model->nr_class;
    int l = model->l;
    fprintf(fp, "nr_class %d\n", nr_class);
    fprintf(fp, "total_sv %d\n",l);

    {
        fprintf(fp, "rho");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->rho[i]);
        fprintf(fp, "\n");
    }

    if(model->label)
    {
        fprintf(fp, "label");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->label[i]);
        fprintf(fp, "\n");
    }

    if(model->probA) // regression has probA only
    {
        fprintf(fp, "probA");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->probA[i]);
        fprintf(fp, "\n");
    }
    if(model->probB)
    {
        fprintf(fp, "probB");
        for(int i=0;i<nr_class*(nr_class-1)/2;i++)
            fprintf(fp," %g",model->probB[i]);
        fprintf(fp, "\n");
    }

    if(model->nSV)
    {
        fprintf(fp, "nr_sv");
        for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->nSV[i]);
        fprintf(fp, "\n");
    }

    fprintf(fp, "SV\n");
    const double * const *sv_coef = model->sv_coef;
    const svm_data *SV = model->SV;

    for(int i=0;i<l;i++)
    {
        for(int j=0;j<nr_class-1;j++)
            fprintf(fp, "%.16g ",sv_coef[j][i]);

        fprintf(fp, "%s\n", SV[i].d->seq_string);
    }

    setlocale(LC_ALL, old_locale);
    free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
    else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,(size_t) max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
static bool read_model_header(FILE *fp, svm_model* model)
{
    svm_parameter& param = model->param;
    char cmd[81];
    param.norc = 0; // initialize this as zero for compatibility
    while(1)
    {
        FSCANF(fp,"%80s",cmd);

        if(strcmp(cmd,"svm_type")==0)
        {
            FSCANF(fp,"%80s",cmd);
            int i;
            for(i=0;svm_type_table[i];i++)
            {
                if(strcmp(svm_type_table[i],cmd)==0)
                {
                    param.svm_type=i;
                    break;
                }
            }
            if(svm_type_table[i] == NULL)
            {
                clog_error(CLOG(LOGGER_ID), "unknown svm type (%s)", cmd);
                return false;
            }
        }
        else if(strcmp(cmd,"kernel_type")==0)
        {
            FSCANF(fp,"%80s",cmd);
            int i;
            for(i=0;kernel_type_table[i];i++)
            {
                if(strcmp(kernel_type_table[i],cmd)==0)
                {
                    param.kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL)
            {
                clog_error(CLOG(LOGGER_ID), "unknown kernel function (%s).", cmd);
                return false;
            }
        }
        else if(strcmp(cmd,"L")==0)
            FSCANF(fp,"%d",&param.L);
        else if(strcmp(cmd,"k")==0)
            FSCANF(fp,"%d",&param.k);
        else if(strcmp(cmd,"d")==0)
            FSCANF(fp,"%d",&param.d);
        else if(strcmp(cmd,"norc")==0)
            FSCANF(fp,"%d",&param.norc);
        else if(strcmp(cmd,"gamma")==0)
            FSCANF(fp,"%lf",&param.gamma);
        else if(strcmp(cmd,"M")==0)
        {
            int tmpM;
            FSCANF(fp,"%d",&tmpM);
            param.M = (uint8_t) tmpM;
        }
        else if(strcmp(cmd,"H")==0)
            FSCANF(fp,"%lf",&param.H);
        else if(strcmp(cmd,"nr_class")==0)
            FSCANF(fp,"%d",&model->nr_class);
        else if(strcmp(cmd,"total_sv")==0)
            FSCANF(fp,"%d",&model->l);
        else if(strcmp(cmd,"rho")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->rho = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->rho[i]);
        }
        else if(strcmp(cmd,"label")==0)
        {
            int n = model->nr_class;
            model->label = Malloc(int,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%d",&model->label[i]);
        }
        else if(strcmp(cmd,"probA")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probA = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->probA[i]);
        }
        else if(strcmp(cmd,"probB")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probB = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->probB[i]);
        }
        else if(strcmp(cmd,"nr_sv")==0)
        {
            int n = model->nr_class;
            model->nSV = Malloc(int,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%d",&model->nSV[i]);
        }
        else if(strcmp(cmd,"SV")==0)
        {
            while(1)
            {
                int c = getc(fp);
                if(c==EOF || c=='\n') break;
            }
            break;
        }
        else
        {
            clog_error(CLOG(LOGGER_ID), "unknown text in model file: [%s]",cmd);
            return false;
        }
    }

    return true;

}

// load a model with gkmtree initialization
svm_model *svm_load_model(const char *model_file_name)
{
    FILE *fp = fopen(model_file_name,"rb");
    if(fp==NULL) return NULL;

    char *old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    // read parameters
    svm_model *model = Malloc(svm_model,1);
    model->rho = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->sv_indices = NULL;
    model->label = NULL;
    model->nSV = NULL;

    // read header
    if (!read_model_header(fp, model))
    {
        clog_error(CLOG(LOGGER_ID), "fscanf failed to read model");
        setlocale(LC_ALL, old_locale);
        free(old_locale);
        free(model->rho);
        free(model->label);
        free(model->nSV);
        free(model);
        return NULL;
    }

    // initialization of gkmkernel after reading header
    gkmkernel_init(&model->param);

    // initialization of SV kmertree
    gkmkernel_init_sv_kmertree_objects(model->nr_class);

    // read sv_coef and SV
    int elements = 0;
    long pos = ftell(fp);

    max_line_len = 1024;
    line = Malloc(char,max_line_len);
    char *p,*endptr,*val;

    while(readline(fp)!=NULL)
    {
        p = strtok(line,":");
        while(1)
        {
            p = strtok(NULL,":");
            if(p == NULL)
                break;
            ++elements;
        }
    }
    elements += model->l;

    fseek(fp,pos,SEEK_SET);

    int m = model->nr_class - 1;
    int l = model->l;
    model->sv_coef = Malloc(double *,m);
    int i;
    for(i=0;i<m;i++)
        model->sv_coef[i] = Malloc(double,l);
    model->SV = Malloc(svm_data,l);

    clog_info(CLOG(LOGGER_ID), "reading %d SVs", l);
    for(i=0;i<l;i++)
    {
        /*
        if ((i > 0) && ((i % 1000) == 0)) {
            clog_info(CLOG(LOGGER_ID), "reading... %d/%d", i, l);
        }
        */

        readline(fp);
        p = strtok(line, " \t");
        model->sv_coef[0][i] = strtod(p,&endptr);
        for(int k=1;k<m;k++)
        {
            p = strtok(NULL, " \t");
            model->sv_coef[k][i] = strtod(p,&endptr);
        }

        val = strtok(NULL,"\n");
        model->SV[i].d = gkmkernel_new_object(val, NULL, i);

        // add SV to kmertree or kmertreecoef
        gkmkernel_add_one_sv(model->SV[i].d, model->sv_coef[0][i], i, model->nr_class);

        // free-up unnecessary data
        gkmkernel_free_object(model->SV[i].d);
    }

    free(line);
    setlocale(LC_ALL, old_locale);
    free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0)
        return NULL;

    model->free_sv = 1; // XXX

    g_sv_svm_data = model->SV;
    g_sv_num = model->l;

    return model;
}

