/*
 * Copyright (c) 2000-2014 Chih-Chung Chang and Chih-Jen Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither name of copyright holders nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Modifications for LS-GKM (W) 2015-2021 Dongwon Lee
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>

#include "libsvm_gkm.h"

#include "clog.h"

int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*((size_t) n));
}
static inline double powi(double base, int times)
{
    double tmp = base, ret = 1.0;

    for(int t=times; t>0; t/=2)
    {
        if(t%2==1) ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
    Cache(int l,long int size);
    ~Cache();

    // request data [0,len)
    // return some position p where [p,len) need to be filled
    // (p >= len if nothing needs to be filled)
    int get_data(const int index, Qfloat **data, int len);
    void swap_index(int i, int j);
private:
    int l;
    long int size;
    struct head_t
    {
        head_t *prev, *next;    // a circular list
        Qfloat *data;
        int len;        // data[0,len) is cached in this entry
    };

    head_t *head;
    head_t lru_head;
    void lru_delete(head_t *h);
    void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
    head = (head_t *)calloc((size_t) l,sizeof(head_t));  // initialized to 0
    size /= sizeof(Qfloat);
    size -= ((size_t) l) * sizeof(head_t) / sizeof(Qfloat);
    size = max(size, 2 * (long int) l); // cache must be large enough for two columns
    lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
    for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
        free(h->data);
    free(head);
}

void Cache::lru_delete(head_t *h)
{
    // delete from current location
    h->prev->next = h->next;
    h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
    // insert to last position
    h->next = &lru_head;
    h->prev = lru_head.prev;
    h->prev->next = h;
    h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
    head_t *h = &head[index];
    if(h->len) lru_delete(h);
    int more = len - h->len;

    if(more > 0)
    {
        // free old space
        while(size < more)
        {
            head_t *old = lru_head.next;
            lru_delete(old);
            free(old->data);
            size += old->len;
            old->data = 0;
            old->len = 0;
        }

        // allocate new space
        h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*((size_t)len));
        size -= more;
        swap(h->len,len);
    }

    lru_insert(h);
    *data = h->data;
    return len;
}

void Cache::swap_index(int i, int j)
{
    if(i==j) return;

    if(head[i].len) lru_delete(&head[i]);
    if(head[j].len) lru_delete(&head[j]);
    swap(head[i].data,head[j].data);
    swap(head[i].len,head[j].len);
    if(head[i].len) lru_insert(&head[i]);
    if(head[j].len) lru_insert(&head[j]);

    if(i>j) swap(i,j);
    for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
    {
        if(h->len > i)
        {
            if(h->len > j)
                swap(h->data[i],h->data[j]);
            else
            {
                // give up
                lru_delete(h);
                free(h->data);
                size += h->len;
                h->data = 0;
                h->len = 0;
            }
        }
    }
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
    virtual Qfloat *get_Q(int column, int len) const = 0;
    virtual double *get_QD() const = 0;
    virtual void swap_index(int i, int j) const = 0;
    virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
    Kernel(int l, svm_data const * x, const svm_parameter& param);
    virtual ~Kernel();

    static double k_function(const svm_data x, const svm_data y);

    virtual Qfloat *get_Q(int column, int len) const = 0;
    virtual double *get_QD() const = 0;
    virtual void swap_index(int i, int j) const // no so const...
    {
        gkmkernel_swap_index(i, j);
        swap(x[i],x[j]);
    }
protected:

    double (Kernel::*kernel_function)(int i, int j) const;
    void kernel_function_batch (int i, int j_start, int j_end) const
    {
        //for test..
        //gkmkernel_kernelfunc_batch(x[i].d, (x+j_start), (j_end-j_start), kvalue);
        //gkmkernel_kernelfunc_batch_all(i, j_start, j_end, temp_kvalue);
        
        if (j_end - j_start < 100) {
            gkmkernel_kernelfunc_batch(x[i].d, (x+j_start), (j_end-j_start), temp_kvalue);
        } else {
            gkmkernel_kernelfunc_batch_all(i, j_start, j_end, temp_kvalue);
        }

        //debug
        /*
        double *temp_kvalue2 = new double [j_end - j_start];
        gkmkernel_kernelfunc_batch(x[i].d, (x+j_start), (j_end-j_start), temp_kvalue2);
        for(int j=0; j<j_end-j_start; j++) {
            if (kvalue[j] != temp_kvalue2[j]) {
                fprintf(stderr, "ERR.[%d %d %d %d]: %f != %f\n", i, j_start, j_end, j, kvalue[j], temp_kvalue2[j]);
            }
        }

        delete [] temp_kvalue2;
        */
    }

    double *temp_kvalue;
private:

    int kernel_type;
    svm_data *x;

    double kernel_gkm(int i, int j) const
    {
        return gkmkernel_kernelfunc(x[i].d, x[j].d);
    }
};

Kernel::Kernel(int l, svm_data const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type)
{
    kernel_function = &Kernel::kernel_gkm;
    clone(x,x_,l);
    gkmkernel_init_problems(x, l); //build kmertree using the entire problem set, which will then be used by gkmkernel_kernelfunc_batch_all
    temp_kvalue = new double [l];
}

Kernel::~Kernel()
{
    delete [] x;
    delete [] temp_kvalue;
    gkmkernel_destroy_problems();
}

double Kernel::k_function(const svm_data x, const svm_data y)
{
    return gkmkernel_kernelfunc(x.d, y.d);
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//  min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//      y^T \alpha = \delta
//      y_i = +1 or -1
//      0 <= alpha_i <= Cp for y_i = 1
//      0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//  Q, p, y, Cp, Cn, and an initial feasible point \alpha
//  l is the size of vectors and matrices
//  eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
    Solver() {};
    virtual ~Solver() {};

    struct SolutionInfo {
        double obj;
        double rho;
        double upper_bound_p;
        double upper_bound_n;
        double r;   // for Solver_NU
    };

    void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
           double *alpha_, double Cp, double Cn, double eps,
           SolutionInfo* si, int shrinking);
protected:
    int active_size;
    schar *y;
    double *G;      // gradient of objective function
    enum { LOWER_BOUND, UPPER_BOUND, FREE };
    char *alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE
    double *alpha;
    const QMatrix *Q;
    const double *QD;
    double eps;
    double Cp,Cn;
    double *p;
    int *active_set;
    double *G_bar;      // gradient, if we treat free variables as 0
    int l;
    bool unshrink;  // XXX

    double get_C(int i)
    {
        return (y[i] > 0)? Cp : Cn;
    }
    void update_alpha_status(int i)
    {
        if(alpha[i] >= get_C(i))
            alpha_status[i] = UPPER_BOUND;
        else if(alpha[i] <= 0)
            alpha_status[i] = LOWER_BOUND;
        else alpha_status[i] = FREE;
    }
    bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
    bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
    bool is_free(int i) { return alpha_status[i] == FREE; }
    void swap_index(int i, int j);
    void reconstruct_gradient();
    virtual int select_working_set(int &i, int &j, double &max_viol);
    virtual double calculate_rho();
    virtual void do_shrinking();
private:
    bool be_shrunk(int i, double Gmax1, double Gmax2);
};

void Solver::swap_index(int i, int j)
{
    Q->swap_index(i,j);
    swap(y[i],y[j]);
    swap(G[i],G[j]);
    swap(alpha_status[i],alpha_status[j]);
    swap(alpha[i],alpha[j]);
    swap(p[i],p[j]);
    swap(active_set[i],active_set[j]);
    swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
    // reconstruct inactive elements of G from G_bar and free variables

    if(active_size == l) return;

    clog_info(CLOG(LOGGER_ID), "reconstruct the whole gradient");

    int i,j;
    int nr_free = 0;

    for(j=active_size;j<l;j++)
        G[j] = G_bar[j] + p[j];

    for(j=0;j<active_size;j++)
        if(is_free(j))
            nr_free++;

    if(2*nr_free < active_size)
        clog_warn(CLOG(LOGGER_ID), "using -h 0 may be faster");

    if (nr_free*l > 2*active_size*(l-active_size))
    {
        for(i=active_size;i<l;i++)
        {
            const Qfloat *Q_i = Q->get_Q(i,active_size);
            for(j=0;j<active_size;j++)
                if(is_free(j))
                    G[i] += alpha[j] * Q_i[j];
        }
    }
    else
    {
        for(i=0;i<active_size;i++)
            if(is_free(i))
            {
                const Qfloat *Q_i = Q->get_Q(i,l);
                double alpha_i = alpha[i];
                for(j=active_size;j<l;j++)
                    G[j] += alpha_i * Q_i[j];
            }
    }
    clog_info(CLOG(LOGGER_ID), "reconstruct done");
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
           double *alpha_, double Cp, double Cn, double eps,
           SolutionInfo* si, int shrinking)
{
    this->l = l;
    this->Q = &Q;
    QD=Q.get_QD();
    clone(p, p_,l);
    clone(y, y_,l);
    clone(alpha,alpha_,l);
    this->Cp = Cp;
    this->Cn = Cn;
    this->eps = eps;
    unshrink = false;

    // initialize alpha_status
    {
        alpha_status = new char[l];
        for(int i=0;i<l;i++)
            update_alpha_status(i);
    }

    // initialize active set (for shrinking)
    {
        active_set = new int[l];
        for(int i=0;i<l;i++)
            active_set[i] = i;
        active_size = l;
    }

    // initialize gradient
    {
        G = new double[l];
        G_bar = new double[l];
        int i;
        for(i=0;i<l;i++)
        {
            G[i] = p[i];
            G_bar[i] = 0;
        }
        for(i=0;i<l;i++)
            if(!is_lower_bound(i))
            {
                const Qfloat *Q_i = Q.get_Q(i,l);
                double alpha_i = alpha[i];
                int j;
                for(j=0;j<l;j++)
                    G[j] += alpha_i*Q_i[j];
                if(is_upper_bound(i))
                    for(j=0;j<l;j++)
                        G_bar[j] += get_C(i) * Q_i[j];
            }
    }

    // optimization step

    int iter = 0;
    int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
    int counter = min(l,1000)+1;
    
    while(iter < max_iter)
    {
        // show progress and do shrinking
        if(--counter == 0)
        {
            counter = min(l,1000);
            if(shrinking) do_shrinking();
        }

        int i,j;
        double max_viol;
        if(select_working_set(i,j,max_viol)!=0)
        {
            clog_debug(CLOG(LOGGER_ID), "max_viol<%g. double check", eps);
            // reconstruct the whole gradient
            reconstruct_gradient();
            // reset active set size and check
            active_size = l;
            clog_info(CLOG(LOGGER_ID), "*");
            if(select_working_set(i,j,max_viol)!=0)
                break;
            else
                counter = 1;    // do shrinking next iteration
        }
        
        ++iter;

        clog_debug(CLOG(LOGGER_ID), "iter %d, max_viol=%g (idx1=%d, idx2=%d)", iter, max_viol, i, j);

        if (iter % 100 == 0) {
            if (clog_get_level(LOGGER_ID) > CLOG_DEBUG) {
                clog_info(CLOG(LOGGER_ID), "iter %d, max_viol=%g", iter, max_viol);
            }
        }

        //clog_debug(CLOG(LOGGER_ID), "selected working-set: index=(%d, %d), grad=(%.6f, %.6f)", i, j, G[i], G[j]);

        // update alpha[i] and alpha[j], handle bounds carefully
        
        const Qfloat *Q_i = Q.get_Q(i,active_size);
        const Qfloat *Q_j = Q.get_Q(j,active_size);

        double C_i = get_C(i);
        double C_j = get_C(j);

        double old_alpha_i = alpha[i];
        double old_alpha_j = alpha[j];

        if(y[i]!=y[j])
        {
            double quad_coef = QD[i]+QD[j]+2*Q_i[j];
            if (quad_coef <= 0)
                quad_coef = TAU;
            double delta = (-G[i]-G[j])/quad_coef;
            double diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;
            
            if(diff > 0)
            {
                if(alpha[j] < 0)
                {
                    alpha[j] = 0;
                    alpha[i] = diff;
                }
            }
            else
            {
                if(alpha[i] < 0)
                {
                    alpha[i] = 0;
                    alpha[j] = -diff;
                }
            }
            if(diff > C_i - C_j)
            {
                if(alpha[i] > C_i)
                {
                    alpha[i] = C_i;
                    alpha[j] = C_i - diff;
                }
            }
            else
            {
                if(alpha[j] > C_j)
                {
                    alpha[j] = C_j;
                    alpha[i] = C_j + diff;
                }
            }
        }
        else
        {
            double quad_coef = QD[i]+QD[j]-2*Q_i[j];
            if (quad_coef <= 0)
                quad_coef = TAU;
            double delta = (G[i]-G[j])/quad_coef;
            double sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;

            if(sum > C_i)
            {
                if(alpha[i] > C_i)
                {
                    alpha[i] = C_i;
                    alpha[j] = sum - C_i;
                }
            }
            else
            {
                if(alpha[j] < 0)
                {
                    alpha[j] = 0;
                    alpha[i] = sum;
                }
            }
            if(sum > C_j)
            {
                if(alpha[j] > C_j)
                {
                    alpha[j] = C_j;
                    alpha[i] = sum - C_j;
                }
            }
            else
            {
                if(alpha[i] < 0)
                {
                    alpha[i] = 0;
                    alpha[j] = sum;
                }
            }
        }

        // update G

        double delta_alpha_i = alpha[i] - old_alpha_i;
        double delta_alpha_j = alpha[j] - old_alpha_j;
        
        for(int k=0;k<active_size;k++)
        {
            G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
        }

        // update alpha_status and G_bar

        {
            bool ui = is_upper_bound(i);
            bool uj = is_upper_bound(j);
            update_alpha_status(i);
            update_alpha_status(j);
            int k;
            if(ui != is_upper_bound(i))
            {
                Q_i = Q.get_Q(i,l);
                if(ui)
                    for(k=0;k<l;k++)
                        G_bar[k] -= C_i * Q_i[k];
                else
                    for(k=0;k<l;k++)
                        G_bar[k] += C_i * Q_i[k];
            }

            if(uj != is_upper_bound(j))
            {
                Q_j = Q.get_Q(j,l);
                if(uj)
                    for(k=0;k<l;k++)
                        G_bar[k] -= C_j * Q_j[k];
                else
                    for(k=0;k<l;k++)
                        G_bar[k] += C_j * Q_j[k];
            }
        }
    }

    if(iter >= max_iter)
    {
        if(active_size < l)
        {
            // reconstruct the whole gradient to calculate objective value
            reconstruct_gradient();
            active_size = l;
        }
        clog_warn(CLOG(LOGGER_ID), "reached max number of iterations (%d)", max_iter);
    }

    // calculate rho

    si->rho = calculate_rho();

    // calculate objective value
    {
        double v = 0;
        int i;
        for(i=0;i<l;i++)
            v += alpha[i] * (G[i] + p[i]);

        si->obj = v/2;
    }

    // put back the solution
    {
        for(int i=0;i<l;i++)
            alpha_[active_set[i]] = alpha[i];
    }

    // juggle everything back
    /*{
        for(int i=0;i<l;i++)
            while(active_set[i] != i)
                swap_index(i,active_set[i]);
                // or Q.swap_index(i,active_set[i]);
    }*/

    si->upper_bound_p = Cp;
    si->upper_bound_n = Cn;

    clog_info(CLOG(LOGGER_ID), "optimization finished, #iter = %d",iter);

    delete[] p;
    delete[] y;
    delete[] alpha;
    delete[] alpha_status;
    delete[] active_set;
    delete[] G;
    delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j, double &max_viol)
{
    // return i,j such that
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: minimizes the decrease of obj value
    //    (if quadratic coefficeint <= 0, replace it with tau)
    //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
    
    double Gmax = -INF;
    double Gmax2 = -INF;
    int Gmax_idx = -1;
    int Gmin_idx = -1;
    double obj_diff_min = INF;

    for(int t=0;t<active_size;t++)
        if(y[t]==+1)    
        {
            if(!is_upper_bound(t))
                if(-G[t] >= Gmax)
                {
                    Gmax = -G[t];
                    Gmax_idx = t;
                }
        }
        else
        {
            if(!is_lower_bound(t))
                if(G[t] >= Gmax)
                {
                    Gmax = G[t];
                    Gmax_idx = t;
                }
        }

    int i = Gmax_idx;
    const Qfloat *Q_i = NULL;
    if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
        Q_i = Q->get_Q(i,active_size);

    for(int j=0;j<active_size;j++)
    {
        if(y[j]==+1)
        {
            if (!is_lower_bound(j))
            {
                double grad_diff=Gmax+G[j];
                if (G[j] >= Gmax2)
                    Gmax2 = G[j];
                if (grad_diff > 0)
                {
                    double obj_diff;
                    double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        else
        {
            if (!is_upper_bound(j))
            {
                double grad_diff= Gmax-G[j];
                if (-G[j] >= Gmax2)
                    Gmax2 = -G[j];
                if (grad_diff > 0)
                {
                    double obj_diff;
                    double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    max_viol = Gmax+Gmax2;
    if(max_viol < eps)
        return 1;

    out_i = Gmax_idx;
    out_j = Gmin_idx;
    return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
    if(is_upper_bound(i))
    {
        if(y[i]==+1)
            return(-G[i] > Gmax1);
        else
            return(-G[i] > Gmax2);
    }
    else if(is_lower_bound(i))
    {
        if(y[i]==+1)
            return(G[i] > Gmax2);
        else    
            return(G[i] > Gmax1);
    }
    else
        return(false);
}

void Solver::do_shrinking()
{
    int i;
    double Gmax1 = -INF;        // max { -y_i * grad(f)_i | i in I_up(\alpha) }
    double Gmax2 = -INF;        // max { y_i * grad(f)_i | i in I_low(\alpha) }

    clog_info(CLOG(LOGGER_ID), "Shrinking");

    // find maximal violating pair first
    for(i=0;i<active_size;i++)
    {
        if(y[i]==+1)    
        {
            if(!is_upper_bound(i))  
            {
                if(-G[i] >= Gmax1)
                    Gmax1 = -G[i];
            }
            if(!is_lower_bound(i))  
            {
                if(G[i] >= Gmax2)
                    Gmax2 = G[i];
            }
        }
        else    
        {
            if(!is_upper_bound(i))  
            {
                if(-G[i] >= Gmax2)
                    Gmax2 = -G[i];
            }
            if(!is_lower_bound(i))  
            {
                if(G[i] >= Gmax1)
                    Gmax1 = G[i];
            }
        }
    }

    if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
    {
        unshrink = true;
        reconstruct_gradient();
        active_size = l;
        clog_info(CLOG(LOGGER_ID), "*");
    }

    for(i=0;i<active_size;i++)
        if (be_shrunk(i, Gmax1, Gmax2))
        {
            active_size--;
            while (active_size > i)
            {
                if (!be_shrunk(active_size, Gmax1, Gmax2))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size--;
            }
        }

    gkmkernel_update_index();

    clog_info(CLOG(LOGGER_ID), "after shrinking, active size is reduced to %d", active_size);
}

double Solver::calculate_rho()
{
    double r;
    int nr_free = 0;
    double ub = INF, lb = -INF, sum_free = 0;
    for(int i=0;i<active_size;i++)
    {
        double yG = y[i]*G[i];

        if(is_upper_bound(i))
        {
            if(y[i]==-1)
                ub = min(ub,yG);
            else
                lb = max(lb,yG);
        }
        else if(is_lower_bound(i))
        {
            if(y[i]==+1)
                ub = min(ub,yG);
            else
                lb = max(lb,yG);
        }
        else
        {
            ++nr_free;
            sum_free += yG;
        }
    }

    if(nr_free>0)
        r = sum_free/nr_free;
    else
        r = (ub+lb)/2;

    return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU: public Solver
{
public:
    Solver_NU() {}
    void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
           double *alpha, double Cp, double Cn, double eps,
           SolutionInfo* si, int shrinking)
    {
        this->si = si;
        Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
    }
private:
    SolutionInfo *si;
    int select_working_set(int &i, int &j, double &max_viol);
    double calculate_rho();
    bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
    void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j, double &max_viol)
{
    // return i,j such that y_i = y_j and
    // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    // j: minimizes the decrease of obj value
    //    (if quadratic coefficeint <= 0, replace it with tau)
    //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

    double Gmaxp = -INF;
    double Gmaxp2 = -INF;
    int Gmaxp_idx = -1;

    double Gmaxn = -INF;
    double Gmaxn2 = -INF;
    int Gmaxn_idx = -1;

    int Gmin_idx = -1;
    double obj_diff_min = INF;

    for(int t=0;t<active_size;t++)
        if(y[t]==+1)
        {
            if(!is_upper_bound(t))
                if(-G[t] >= Gmaxp)
                {
                    Gmaxp = -G[t];
                    Gmaxp_idx = t;
                }
        }
        else
        {
            if(!is_lower_bound(t))
                if(G[t] >= Gmaxn)
                {
                    Gmaxn = G[t];
                    Gmaxn_idx = t;
                }
        }

    int ip = Gmaxp_idx;
    int in = Gmaxn_idx;
    const Qfloat *Q_ip = NULL;
    const Qfloat *Q_in = NULL;
    if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
        Q_ip = Q->get_Q(ip,active_size);
    if(in != -1)
        Q_in = Q->get_Q(in,active_size);

    for(int j=0;j<active_size;j++)
    {
        if(y[j]==+1)
        {
            if (!is_lower_bound(j)) 
            {
                double grad_diff=Gmaxp+G[j];
                if (G[j] >= Gmaxp2)
                    Gmaxp2 = G[j];
                if (grad_diff > 0)
                {
                    double obj_diff;
                    double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
        else
        {
            if (!is_upper_bound(j))
            {
                double grad_diff=Gmaxn-G[j];
                if (-G[j] >= Gmaxn2)
                    Gmaxn2 = -G[j];
                if (grad_diff > 0)
                {
                    double obj_diff;
                    double quad_coef = QD[in]+QD[j]-2*Q_in[j];
                    if (quad_coef > 0)
                        obj_diff = -(grad_diff*grad_diff)/quad_coef;
                    else
                        obj_diff = -(grad_diff*grad_diff)/TAU;

                    if (obj_diff <= obj_diff_min)
                    {
                        Gmin_idx=j;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }
    }

    max_viol = max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2);

    if(max_viol < eps)
        return 1;

    if (y[Gmin_idx] == +1)
        out_i = Gmaxp_idx;
    else
        out_i = Gmaxn_idx;
    out_j = Gmin_idx;

    return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
    if(is_upper_bound(i))
    {
        if(y[i]==+1)
            return(-G[i] > Gmax1);
        else    
            return(-G[i] > Gmax4);
    }
    else if(is_lower_bound(i))
    {
        if(y[i]==+1)
            return(G[i] > Gmax2);
        else    
            return(G[i] > Gmax3);
    }
    else
        return(false);
}

void Solver_NU::do_shrinking()
{
    double Gmax1 = -INF;    // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
    double Gmax2 = -INF;    // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
    double Gmax3 = -INF;    // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
    double Gmax4 = -INF;    // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

    clog_info(CLOG(LOGGER_ID), "Shrinking");

    // find maximal violating pair first
    int i;
    for(i=0;i<active_size;i++)
    {
        if(!is_upper_bound(i))
        {
            if(y[i]==+1)
            {
                if(-G[i] > Gmax1) Gmax1 = -G[i];
            }
            else    if(-G[i] > Gmax4) Gmax4 = -G[i];
        }
        if(!is_lower_bound(i))
        {
            if(y[i]==+1)
            {   
                if(G[i] > Gmax2) Gmax2 = G[i];
            }
            else    if(G[i] > Gmax3) Gmax3 = G[i];
        }
    }

    if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
    {
        unshrink = true;
        reconstruct_gradient();
        active_size = l;
    }

    for(i=0;i<active_size;i++)
        if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
        {
            active_size--;
            while (active_size > i)
            {
                if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
                {
                    swap_index(i,active_size);
                    break;
                }
                active_size--;
            }
        }

    gkmkernel_update_index();

    clog_info(CLOG(LOGGER_ID), "after shrinking, active size is reduced to %d", active_size);
}

double Solver_NU::calculate_rho()
{
    int nr_free1 = 0,nr_free2 = 0;
    double ub1 = INF, ub2 = INF;
    double lb1 = -INF, lb2 = -INF;
    double sum_free1 = 0, sum_free2 = 0;

    for(int i=0;i<active_size;i++)
    {
        if(y[i]==+1)
        {
            if(is_upper_bound(i))
                lb1 = max(lb1,G[i]);
            else if(is_lower_bound(i))
                ub1 = min(ub1,G[i]);
            else
            {
                ++nr_free1;
                sum_free1 += G[i];
            }
        }
        else
        {
            if(is_upper_bound(i))
                lb2 = max(lb2,G[i]);
            else if(is_lower_bound(i))
                ub2 = min(ub2,G[i]);
            else
            {
                ++nr_free2;
                sum_free2 += G[i];
            }
        }
    }

    double r1,r2;
    if(nr_free1 > 0)
        r1 = sum_free1/nr_free1;
    else
        r1 = (ub1+lb1)/2;
    
    if(nr_free2 > 0)
        r2 = sum_free2/nr_free2;
    else
        r2 = (ub2+lb2)/2;
    
    si->r = (r1+r2)/2;
    return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
    SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
    :Kernel(prob.l, prob.x, param)
    {
        clone(y,y_,prob.l);
        cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
        QD = new double[prob.l];
        for(int i=0;i<prob.l;i++)
            QD[i] = (this->*kernel_function)(i,i);
    }
    
    Qfloat *get_Q(int i, int len) const
    {
        Qfloat *data;
        int start, j;
        if((start = cache->get_data(i,&data,len)) < len)
        {
            this->kernel_function_batch(i, start, len);
            for(j=start;j<len;j++)
                data[j] = (Qfloat)(y[i]*y[j]*temp_kvalue[j-start]);
        }
        return data;
    }

    double *get_QD() const
    {
        return QD;
    }

    void swap_index(int i, int j) const
    {
        cache->swap_index(i,j);
        Kernel::swap_index(i,j);
        swap(y[i],y[j]);
        swap(QD[i],QD[j]);
    }

    ~SVC_Q()
    {
        delete[] y;
        delete cache;
        delete[] QD;
    }
private:
    schar *y;
    Cache *cache;
    double *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
    ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
    :Kernel(prob.l, prob.x, param)
    {
        cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
        QD = new double[prob.l];
        for(int i=0;i<prob.l;i++)
            QD[i] = (this->*kernel_function)(i,i);
    }
    
    Qfloat *get_Q(int i, int len) const
    {
        Qfloat *data;
        int start, j;
        if((start = cache->get_data(i,&data,len)) < len)
        {
            this->kernel_function_batch(i, start, len);
            for(j=start;j<len;j++)
                data[j] = (Qfloat)(temp_kvalue[j-start]);
        }
        return data;
    }

    double *get_QD() const
    {
        return QD;
    }

    void swap_index(int i, int j) const
    {
        cache->swap_index(i,j);
        Kernel::swap_index(i,j);
        swap(QD[i],QD[j]);
    }

    ~ONE_CLASS_Q()
    {
        delete cache;
        delete[] QD;
    }
private:
    Cache *cache;
    double *QD;
};

class SVR_Q: public Kernel
{ 
public:
    SVR_Q(const svm_problem& prob, const svm_parameter& param)
    :Kernel(prob.l, prob.x, param)
    {
        l = prob.l;
        cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
        QD = new double[2*l];
        sign = new schar[2*l];
        index = new int[2*l];
        for(int k=0;k<l;k++)
        {
            sign[k] = 1;
            sign[k+l] = -1;
            index[k] = k;
            index[k+l] = k;
            QD[k] = (this->*kernel_function)(k,k);
            QD[k+l] = QD[k];
        }
        buffer[0] = new Qfloat[2*l];
        buffer[1] = new Qfloat[2*l];
        next_buffer = 0;
    }

    void swap_index(int i, int j) const
    {
        swap(sign[i],sign[j]);
        swap(index[i],index[j]);
        swap(QD[i],QD[j]);
    }
    
    Qfloat *get_Q(int i, int len) const
    {
        Qfloat *data;
        int j, real_i = index[i];
        if(cache->get_data(real_i,&data,l) < l)
        {
            this->kernel_function_batch(real_i, 0, l);
            for(j=0;j<l;j++)
                data[j] = (Qfloat)(temp_kvalue[j]);
        }

        // reorder and copy
        Qfloat *buf = buffer[next_buffer];
        next_buffer = 1 - next_buffer;
        schar si = sign[i];
        for(j=0;j<len;j++)
            buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
        return buf;
    }

    double *get_QD() const
    {
        return QD;
    }

    ~SVR_Q()
    {
        delete cache;
        delete[] sign;
        delete[] index;
        delete[] buffer[0];
        delete[] buffer[1];
        delete[] QD;
    }
private:
    int l;
    Cache *cache;
    schar *sign;
    int *index;
    mutable int next_buffer;
    Qfloat *buffer[2];
    double *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(
    const svm_problem *prob, const svm_parameter* param,
    double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
    int l = prob->l;
    double *minus_ones = new double[l];
    schar *y = new schar[l];

    int i;

    for(i=0;i<l;i++)
    {
        alpha[i] = 0;
        minus_ones[i] = -1;
        if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
    }

    Solver s;
    s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
        alpha, Cp, Cn, param->eps, si, param->shrinking);

    double sum_alpha=0;
    for(i=0;i<l;i++)
        sum_alpha += alpha[i];

    if (Cp==Cn)
        clog_info(CLOG(LOGGER_ID), "nu = %f", sum_alpha/(Cp*prob->l));

    for(i=0;i<l;i++)
        alpha[i] *= y[i];

    delete[] minus_ones;
    delete[] y;
}

static void solve_nu_svc(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int i;
    int l = prob->l;
    double nu = param->nu;

    schar *y = new schar[l];

    for(i=0;i<l;i++)
        if(prob->y[i]>0)
            y[i] = +1;
        else
            y[i] = -1;

    double sum_pos = nu*l/2;
    double sum_neg = nu*l/2;

    for(i=0;i<l;i++)
        if(y[i] == +1)
        {
            alpha[i] = min(1.0,sum_pos);
            sum_pos -= alpha[i];
        }
        else
        {
            alpha[i] = min(1.0,sum_neg);
            sum_neg -= alpha[i];
        }

    double *zeros = new double[l];

    for(i=0;i<l;i++)
        zeros[i] = 0;

    Solver_NU s;
    s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
        alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
    double r = si->r;

    clog_info(CLOG(LOGGER_ID), "C = %f",1/r);

    for(i=0;i<l;i++)
        alpha[i] *= y[i]/r;

    si->rho /= r;
    si->obj /= (r*r);
    si->upper_bound_p = 1/r;
    si->upper_bound_n = 1/r;

    delete[] y;
    delete[] zeros;
}

static void solve_one_class(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double *zeros = new double[l];
    schar *ones = new schar[l];
    int i;

    int n = (int)(param->nu*prob->l);   // # of alpha's at upper bound

    for(i=0;i<n;i++)
        alpha[i] = 1;
    if(n<prob->l)
        alpha[n] = param->nu * prob->l - n;
    for(i=n+1;i<l;i++)
        alpha[i] = 0;

    for(i=0;i<l;i++)
    {
        zeros[i] = 0;
        ones[i] = 1;
    }

    Solver s;
    s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
        alpha, 1.0, 1.0, param->eps, si, param->shrinking);

    delete[] zeros;
    delete[] ones;
}

static void solve_epsilon_svr(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    int i;

    for(i=0;i<l;i++)
    {
        alpha2[i] = 0;
        linear_term[i] = param->p - prob->y[i];
        y[i] = 1;

        alpha2[i+l] = 0;
        linear_term[i+l] = param->p + prob->y[i];
        y[i+l] = -1;
    }

    Solver s;
    s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
        alpha2, param->C, param->C, param->eps, si, param->shrinking);

    double sum_alpha = 0;
    for(i=0;i<l;i++)
    {
        alpha[i] = alpha2[i] - alpha2[i+l];
        sum_alpha += fabs(alpha[i]);
    }
    clog_info(CLOG(LOGGER_ID), "nu = %f",sum_alpha/(param->C*l));

    delete[] alpha2;
    delete[] linear_term;
    delete[] y;
}

static void solve_nu_svr(
    const svm_problem *prob, const svm_parameter *param,
    double *alpha, Solver::SolutionInfo* si)
{
    int l = prob->l;
    double C = param->C;
    double *alpha2 = new double[2*l];
    double *linear_term = new double[2*l];
    schar *y = new schar[2*l];
    int i;

    double sum = C * param->nu * l / 2;
    for(i=0;i<l;i++)
    {
        alpha2[i] = alpha2[i+l] = min(sum,C);
        sum -= alpha2[i];

        linear_term[i] = - prob->y[i];
        y[i] = 1;

        linear_term[i+l] = prob->y[i];
        y[i+l] = -1;
    }

    Solver_NU s;
    s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
        alpha2, C, C, param->eps, si, param->shrinking);

    clog_info(CLOG(LOGGER_ID), "epsilon = %f",-si->r);

    for(i=0;i<l;i++)
        alpha[i] = alpha2[i] - alpha2[i+l];

    delete[] alpha2;
    delete[] linear_term;
    delete[] y;
}

//
// decision_function
//
struct decision_function
{
    double *alpha;
    double rho;
};

static decision_function svm_train_one(
    const svm_problem *prob, const svm_parameter *param,
    double Cp, double Cn)
{
    double *alpha = Malloc(double,prob->l);
    Solver::SolutionInfo si;

    clog_info(CLOG(LOGGER_ID), "begin SVM training");

    switch(param->svm_type)
    {
        case C_SVC:
            solve_c_svc(prob,param,alpha,&si,Cp,Cn);
            break;
        case NU_SVC:
            solve_nu_svc(prob,param,alpha,&si);
            break;
        case ONE_CLASS:
            solve_one_class(prob,param,alpha,&si);
            break;
        case EPSILON_SVR:
            solve_epsilon_svr(prob,param,alpha,&si);
            break;
        case NU_SVR:
            solve_nu_svr(prob,param,alpha,&si);
            break;
    }

    clog_info(CLOG(LOGGER_ID), "obj = %f, rho = %f",si.obj,si.rho);

    // output SVs

    int nSV = 0;
    int nBSV = 0;
    for(int i=0;i<prob->l;i++)
    {
        if(fabs(alpha[i]) > 0)
        {
            ++nSV;
            if(prob->y[i] > 0)
            {
                if(fabs(alpha[i]) >= si.upper_bound_p)
                    ++nBSV;
            }
            else
            {
                if(fabs(alpha[i]) >= si.upper_bound_n)
                    ++nBSV;
            }
        }
    }

    clog_info(CLOG(LOGGER_ID), "nSV = %d, nBSV = %d",nSV,nBSV);

    decision_function f;
    f.alpha = alpha;
    f.rho = si.rho;
    return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
    int l, const double *dec_values, const double *labels, 
    double& A, double& B)
{
    double prior1=0, prior0 = 0;
    int i;

    for (i=0;i<l;i++)
        if (labels[i] > 0) prior1+=1;
        else prior0+=1;
    
    int max_iter=100;   // Maximal number of iterations
    double min_step=1e-10;  // Minimal step taken in line search
    double sigma=1e-12; // For numerically strict PD of Hessian
    double eps=1e-5;
    double hiTarget=(prior1+1.0)/(prior1+2.0);
    double loTarget=1/(prior0+2.0);
    double *t=Malloc(double,l);
    double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
    double newA,newB,newf,d1,d2;
    int iter;
    
    // Initial Point and Initial Fun Value
    A=0.0; B=log((prior0+1.0)/(prior1+1.0));
    double fval = 0.0;

    for (i=0;i<l;i++)
    {
        if (labels[i]>0) t[i]=hiTarget;
        else t[i]=loTarget;
        fApB = dec_values[i]*A+B;
        if (fApB>=0)
            fval += t[i]*fApB + log(1+exp(-fApB));
        else
            fval += (t[i] - 1)*fApB +log(1+exp(fApB));
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11=sigma; // numerically ensures strict PD
        h22=sigma;
        h21=0.0;g1=0.0;g2=0.0;
        for (i=0;i<l;i++)
        {
            fApB = dec_values[i]*A+B;
            if (fApB >= 0)
            {
                p=exp(-fApB)/(1.0+exp(-fApB));
                q=1.0/(1.0+exp(-fApB));
            }
            else
            {
                p=1.0/(1.0+exp(fApB));
                q=exp(fApB)/(1.0+exp(fApB));
            }
            d2=p*q;
            h11+=dec_values[i]*dec_values[i]*d2;
            h22+=d2;
            h21+=dec_values[i]*d2;
            d1=t[i]-p;
            g1+=dec_values[i]*d1;
            g2+=d1;
        }

        // Stopping Criteria
        if (fabs(g1)<eps && fabs(g2)<eps)
            break;

        // Finding Newton direction: -inv(H') * g
        det=h11*h22-h21*h21;
        dA=-(h22*g1 - h21 * g2) / det;
        dB=-(-h21*g1+ h11 * g2) / det;
        gd=g1*dA+g2*dB;


        stepsize = 1;       // Line Search
        while (stepsize >= min_step)
        {
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            // New function value
            newf = 0.0;
            for (i=0;i<l;i++)
            {
                fApB = dec_values[i]*newA+newB;
                if (fApB >= 0)
                    newf += t[i]*fApB + log(1+exp(-fApB));
                else
                    newf += (t[i] - 1)*fApB +log(1+exp(fApB));
            }
            // Check sufficient decrease
            if (newf<fval+0.0001*stepsize*gd)
            {
                A=newA;B=newB;fval=newf;
                break;
            }
            else
                stepsize = stepsize / 2.0;
        }

        if (stepsize < min_step)
        {
            clog_warn(CLOG(LOGGER_ID), "Line search fails in two-class probability estimates");
            break;
        }
    }

    if (iter>=max_iter)
        clog_warn(CLOG(LOGGER_ID), "Reaching maximal iterations in two-class probability estimates");
    free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
    double fApB = decision_value*A+B;
    // 1-p used later; avoid catastrophic cancellation
    if (fApB >= 0)
        return exp(-fApB)/(1.0+exp(-fApB));
    else
        return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
    int t,j;
    int iter = 0, max_iter=max(100,k);
    double **Q=Malloc(double *,k);
    double *Qp=Malloc(double,k);
    double pQp, eps=0.005/k;
    
    for (t=0;t<k;t++)
    {
        p[t]=1.0/k;  // Valid if k = 1
        Q[t]=Malloc(double,k);
        Q[t][t]=0;
        for (j=0;j<t;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=Q[j][t];
        }
        for (j=t+1;j<k;j++)
        {
            Q[t][t]+=r[j][t]*r[j][t];
            Q[t][j]=-r[j][t]*r[t][j];
        }
    }
    for (iter=0;iter<max_iter;iter++)
    {
        // stopping condition, recalculate QP,pQP for numerical accuracy
        pQp=0;
        for (t=0;t<k;t++)
        {
            Qp[t]=0;
            for (j=0;j<k;j++)
                Qp[t]+=Q[t][j]*p[j];
            pQp+=p[t]*Qp[t];
        }
        double max_error=0;
        for (t=0;t<k;t++)
        {
            double error=fabs(Qp[t]-pQp);
            if (error>max_error)
                max_error=error;
        }
        if (max_error<eps) break;
        
        for (t=0;t<k;t++)
        {
            double diff=(-Qp[t]+pQp)/Q[t][t];
            p[t]+=diff;
            pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
            for (j=0;j<k;j++)
            {
                Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
                p[j]/=(1+diff);
            }
        }
    }
    if (iter>=max_iter)
        clog_warn(CLOG(LOGGER_ID), "Exceeds max_iter in multiclass_prob");
    for(t=0;t<k;t++) free(Q[t]);
    free(Q);
    free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
    const svm_problem *prob, const svm_parameter *param,
    double Cp, double Cn, double& probA, double& probB)
{
    int i;
    int nr_fold = 5;
    int *perm = Malloc(int,prob->l);
    double *dec_values = Malloc(double,prob->l);

    clog_info(CLOG(LOGGER_ID), "perform %d-fold CV to estimate sigmoid parameters", nr_fold);

    // random shuffle
    for(i=0;i<prob->l;i++) perm[i]=i;
    for(i=0;i<prob->l;i++)
    {
        int j = i+rand()%(prob->l-i);
        swap(perm[i],perm[j]);
    }
    for(i=0;i<nr_fold;i++)
    {
        int begin = i*prob->l/nr_fold;
        int end = (i+1)*prob->l/nr_fold;
        int j,k;
        struct svm_problem subprob;

        clog_info(CLOG(LOGGER_ID), "Cross-validation (%d/%d)", i+1, nr_fold);

        subprob.l = prob->l-(end-begin);
        subprob.x = Malloc(union svm_data,subprob.l);
        subprob.y = Malloc(double,subprob.l);
            
        k=0;
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end;j<prob->l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        int p_count=0,n_count=0;
        for(j=0;j<k;j++)
            if(subprob.y[j]>0)
                p_count++;
            else
                n_count++;

        if(p_count==0 && n_count==0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 0;
        else if(p_count > 0 && n_count == 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = 1;
        else if(p_count == 0 && n_count > 0)
            for(j=begin;j<end;j++)
                dec_values[perm[j]] = -1;
        else
        {
            svm_parameter subparam = *param;
            subparam.probability=0;
            subparam.C=1.0;
            subparam.nr_weight=2;
            subparam.weight_label = Malloc(int,2);
            subparam.weight = Malloc(double,2);
            subparam.weight_label[0]=+1;
            subparam.weight_label[1]=-1;
            subparam.weight[0]=Cp;
            subparam.weight[1]=Cn;
            struct svm_model *submodel = svm_train(&subprob,&subparam);

            gkmkernel_init_sv(submodel->SV, submodel->sv_coef[0], submodel->nr_class, submodel->l);
            for(j=begin;j<end;j++)
            {
                svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]]));
                // ensure +1 -1 order; reason not using CV subroutine
                dec_values[perm[j]] *= submodel->label[0];
            }       
            gkmkernel_destroy_sv();
            svm_free_and_destroy_model(&submodel);
            svm_destroy_param(&subparam);
        }
        free(subprob.x);
        free(subprob.y);
    }       
    sigmoid_train(prob->l,dec_values,prob->y,probA,probB);

    clog_info(CLOG(LOGGER_ID), "estimation of sigmoid parameters finished");
    free(dec_values);
    free(perm);

}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
    const svm_problem *prob, const svm_parameter *param)
{
    int i;
    int nr_fold = 5;
    double *ymv = Malloc(double,prob->l);
    double mae = 0;

    svm_parameter newparam = *param;
    newparam.probability = 0;
    svm_cross_validation(prob,&newparam,nr_fold,0,ymv,NULL);
    for(i=0;i<prob->l;i++)
    {
        ymv[i]=prob->y[i]-ymv[i];
        mae += fabs(ymv[i]);
    }       
    mae /= prob->l;
    double std=sqrt(2*mae*mae);
    int count=0;
    mae=0;
    for(i=0;i<prob->l;i++)
        if (fabs(ymv[i]) > 5*std) 
            count=count+1;
        else 
            mae+=fabs(ymv[i]);
    mae /= (prob->l-count);
    clog_info(CLOG(LOGGER_ID), "Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g",mae);
    free(ymv);
    return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
    int l = prob->l;
    int max_nr_class = 16;
    int nr_class = 0;
    int *label = Malloc(int,max_nr_class);
    int *count = Malloc(int,max_nr_class);
    int *data_label = Malloc(int,l);
    int i;

    for(i=0;i<l;i++)
    {
        int this_label = (int)prob->y[i];
        int j;
        for(j=0;j<nr_class;j++)
        {
            if(this_label == label[j])
            {
                ++count[j];
                break;
            }
        }
        data_label[i] = j;
        if(j == nr_class)
        {
            if(nr_class == max_nr_class)
            {
                max_nr_class *= 2;
                label = (int *)realloc(label,((size_t) max_nr_class)*sizeof(int));
                count = (int *)realloc(count,((size_t) max_nr_class)*sizeof(int));
            }
            label[nr_class] = this_label;
            count[nr_class] = 1;
            ++nr_class;
        }
    }

    //
    // Labels are ordered by their first occurrence in the training set. 
    // However, for two-class sets with -1/+1 labels and -1 appears first, 
    // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
    //
    if (nr_class == 2 && label[0] == -1 && label[1] == 1)
    {
        swap(label[0],label[1]);
        swap(count[0],count[1]);
        for(i=0;i<l;i++)
        {
            if(data_label[i] == 0)
                data_label[i] = 1;
            else
                data_label[i] = 0;
        }
    }

    int *start = Malloc(int,nr_class);
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];
    for(i=0;i<l;i++)
    {
        perm[start[data_label[i]]] = i;
        ++start[data_label[i]];
    }
    start[0] = 0;
    for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+count[i-1];

    *nr_class_ret = nr_class;
    *label_ret = label;
    *start_ret = start;
    *count_ret = count;
    free(data_label);
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
    svm_model *model = Malloc(svm_model,1);
    model->param = *param;
    model->free_sv = 0; // XXX

    if(param->svm_type == ONE_CLASS ||
       param->svm_type == EPSILON_SVR ||
       param->svm_type == NU_SVR)
    {
        // regression or one-class-svm
        model->nr_class = 2;
        model->label = NULL;
        model->nSV = NULL;
        model->probA = NULL; model->probB = NULL;
        model->sv_coef = Malloc(double *,1);

        if(param->probability && 
           (param->svm_type == EPSILON_SVR ||
            param->svm_type == NU_SVR))
        {
            model->probA = Malloc(double,1);
            model->probA[0] = svm_svr_probability(prob,param);
        }

        decision_function f = svm_train_one(prob,param,0,0);
        model->rho = Malloc(double,1);
        model->rho[0] = f.rho;

        int nSV = 0;
        int i;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0) ++nSV;
        model->l = nSV;
        model->SV = Malloc(svm_data,nSV);
        model->sv_coef[0] = Malloc(double,nSV);
        model->sv_indices = Malloc(int,nSV);
        int j = 0;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0)
            {
                model->SV[j] = prob->x[i];
                model->sv_coef[0][j] = f.alpha[i];
                model->sv_indices[j] = i+1;
                ++j;
            }       

        free(f.alpha);
    }
    else
    {
        // classification
        int l = prob->l;
        int nr_class;
        int *label = NULL;
        int *start = NULL;
        int *count = NULL;
        int *perm = Malloc(int,l);

        // group training data of the same class
        svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
        if(nr_class == 1) 
            clog_warn(CLOG(LOGGER_ID), "training data in only one class. See README for details.");
        
        svm_data *x = Malloc(svm_data,l);
        int i;
        for(i=0;i<l;i++)
            x[i] = prob->x[perm[i]];

        // calculate weighted C

        double *weighted_C = Malloc(double, nr_class);
        for(i=0;i<nr_class;i++)
            weighted_C[i] = param->C;
        for(i=0;i<param->nr_weight;i++)
        {   
            int j;
            for(j=0;j<nr_class;j++)
                if(param->weight_label[i] == label[j])
                    break;
            if(j == nr_class)
                clog_warn(CLOG(LOGGER_ID), "class label %d specified in weight is not found", param->weight_label[i]);
            else
                weighted_C[j] *= param->weight[i];
        }

        // train k*(k-1)/2 models
        
        bool *nonzero = Malloc(bool,l);
        for(i=0;i<l;i++)
            nonzero[i] = false;
        decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

        double *probA=NULL,*probB=NULL;
        if (param->probability)
        {
            probA=Malloc(double,nr_class*(nr_class-1)/2);
            probB=Malloc(double,nr_class*(nr_class-1)/2);
        }

        int p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                svm_problem sub_prob;
                int si = start[i], sj = start[j];
                int ci = count[i], cj = count[j];
                sub_prob.l = ci+cj;
                sub_prob.x = Malloc(svm_data,sub_prob.l);
                sub_prob.y = Malloc(double,sub_prob.l);
                int k;
                for(k=0;k<ci;k++)
                {
                    sub_prob.x[k] = x[si+k];
                    sub_prob.y[k] = +1;
                }
                for(k=0;k<cj;k++)
                {
                    sub_prob.x[ci+k] = x[sj+k];
                    sub_prob.y[ci+k] = -1;
                }

                if(param->probability)
                    svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

                f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
                for(k=0;k<ci;k++)
                    if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
                        nonzero[si+k] = true;
                for(k=0;k<cj;k++)
                    if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
                        nonzero[sj+k] = true;
                free(sub_prob.x);
                free(sub_prob.y);
                ++p;
            }

        // build output

        model->nr_class = nr_class;
        
        model->label = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            model->label[i] = label[i];
        
        model->rho = Malloc(double,nr_class*(nr_class-1)/2);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            model->rho[i] = f[i].rho;

        if(param->probability)
        {
            model->probA = Malloc(double,nr_class*(nr_class-1)/2);
            model->probB = Malloc(double,nr_class*(nr_class-1)/2);
            for(i=0;i<nr_class*(nr_class-1)/2;i++)
            {
                model->probA[i] = probA[i];
                model->probB[i] = probB[i];
            }
        }
        else
        {
            model->probA=NULL;
            model->probB=NULL;
        }

        int total_sv = 0;
        int *nz_count = Malloc(int,nr_class);
        model->nSV = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
        {
            int nSV = 0;
            for(int j=0;j<count[i];j++)
                if(nonzero[start[i]+j])
                {   
                    ++nSV;
                    ++total_sv;
                }
            model->nSV[i] = nSV;
            nz_count[i] = nSV;
        }
        
        clog_info(CLOG(LOGGER_ID), "Total nSV = %d",total_sv);

        model->l = total_sv;
        model->SV = Malloc(svm_data,total_sv);
        model->sv_indices = Malloc(int,total_sv);
        p = 0;
        for(i=0;i<l;i++)
            if(nonzero[i])
            {
                model->SV[p] = x[i];
                model->sv_indices[p++] = perm[i] + 1;
            }

        int *nz_start = Malloc(int,nr_class);
        nz_start[0] = 0;
        for(i=1;i<nr_class;i++)
            nz_start[i] = nz_start[i-1]+nz_count[i-1];

        model->sv_coef = Malloc(double *,nr_class-1);
        for(i=0;i<nr_class-1;i++)
            model->sv_coef[i] = Malloc(double,total_sv);

        p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

                int si = start[i];
                int sj = start[j];
                int ci = count[i];
                int cj = count[j];
                
                int q = nz_start[i];
                int k;
                for(k=0;k<ci;k++)
                    if(nonzero[si+k])
                        model->sv_coef[j-1][q++] = f[p].alpha[k];
                q = nz_start[j];
                for(k=0;k<cj;k++)
                    if(nonzero[sj+k])
                        model->sv_coef[i][q++] = f[p].alpha[ci+k];
                ++p;
            }
        
        free(label);
        free(probA);
        free(probB);
        free(count);
        free(perm);
        free(start);
        free(x);
        free(weighted_C);
        free(nonzero);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            free(f[i].alpha);
        free(f);
        free(nz_count);
        free(nz_start);
    }
    return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, int icv, double *target, const char *filename)
{
    int i;
    int *fold_start;
    int l = prob->l;
    int *perm = Malloc(int,l);
    int nr_class;
    if (nr_fold > l)
    {
        nr_fold = l;
        clog_warn(CLOG(LOGGER_ID), "# folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)");
    }

    clog_info(CLOG(LOGGER_ID), "Perform %d-fold Cross-Validation", nr_fold);

    fold_start = Malloc(int,nr_fold+1);
    // stratified cv may not give leave-one-out rate
    // Each class to l folds -> some folds may have zero elements
    if((param->svm_type == C_SVC ||
        param->svm_type == NU_SVC) && nr_fold < l)
    {
        int *start = NULL;
        int *label = NULL;
        int *count = NULL;
        svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

        // random shuffle and then data grouped by fold using the array perm
        int *fold_count = Malloc(int,nr_fold);
        int c;
        int *index = Malloc(int,l);
        for(i=0;i<l;i++)
            index[i]=perm[i];
        for (c=0; c<nr_class; c++) 
            for(i=0;i<count[c];i++)
            {
                int j = i+rand()%(count[c]-i);
                swap(index[start[c]+j],index[start[c]+i]);
            }
        for(i=0;i<nr_fold;i++)
        {
            fold_count[i] = 0;
            for (c=0; c<nr_class;c++)
                fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
        }
        fold_start[0]=0;
        for (i=1;i<=nr_fold;i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        for (c=0; c<nr_class;c++)
            for(i=0;i<nr_fold;i++)
            {
                int begin = start[c]+i*count[c]/nr_fold;
                int end = start[c]+(i+1)*count[c]/nr_fold;
                for(int j=begin;j<end;j++)
                {
                    perm[fold_start[i]] = index[j];
                    fold_start[i]++;
                }
            }
        fold_start[0]=0;
        for (i=1;i<=nr_fold;i++)
            fold_start[i] = fold_start[i-1]+fold_count[i-1];
        free(start);
        free(label);
        free(count);
        free(index);
        free(fold_count);
    }
    else
    {
        for(i=0;i<l;i++) perm[i]=i;
        for(i=0;i<l;i++)
        {
            int j = i+rand()%(l-i);
            swap(perm[i],perm[j]);
        }
        for(i=0;i<=nr_fold;i++)
            fold_start[i]=i*l/nr_fold;
    }

    //store all decision values
    int num_dec_values = 1;
    double *dec_values;
    int *cvset;

    if(param->svm_type == C_SVC || param->svm_type == NU_SVC)
        num_dec_values = nr_class*(nr_class-1)/2;

    dec_values = Malloc(double, l*num_dec_values);
    cvset = Malloc(int, l);
    for(i=0; i<l; i++) { cvset[i] = 0; }

    for(i=0;i<nr_fold;i++)
    {
        int begin = fold_start[i];
        int end = fold_start[i+1];
        int j,k;
        struct svm_problem subprob;

        //if icv is given, run the i-th CV only
        if ((icv > 0) && (icv != (i+1))) {
            continue;
        }

        clog_info(CLOG(LOGGER_ID), "Cross-Validation (%d/%d)", i+1, nr_fold);

        subprob.l = l-(end-begin);
        subprob.x = Malloc(union svm_data,subprob.l);
        subprob.y = Malloc(double,subprob.l);
            
        k=0;
        for(j=0;j<begin;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for(j=end;j<l;j++)
        {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }

        struct svm_model *submodel = svm_train(&subprob,param);

        gkmkernel_init_sv(submodel->SV, submodel->sv_coef[0], submodel->nr_class, submodel->l);

        if(param->probability && 
           (param->svm_type == C_SVC || param->svm_type == NU_SVC))
        {
            double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
            for(j=begin;j<end;j++) 
            {
                cvset[perm[j]] = i + 1;
                target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
            }
            free(prob_estimates);
        }
        else
        {
            for(j=begin;j<end;j++) 
            {
                cvset[perm[j]] = i + 1;
                target[perm[j]] = svm_predict_values(submodel, prob->x[perm[j]], (dec_values + (perm[j]*num_dec_values)));
                //target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
            }
        }

        gkmkernel_destroy_sv();
        svm_free_and_destroy_model(&submodel);

        free(subprob.x);
        free(subprob.y);
    }       

    if (filename)
    {
        clog_info(CLOG(LOGGER_ID), "write CV results to %s", filename);
        FILE *fp = fopen(filename,"w");
        if(fp==NULL)
        {
            clog_error(CLOG(LOGGER_ID), "cannot open file %s", filename);
        }
        else 
        {
            int j;
            for (i=0; i<prob->l; i++)
            {
                //if icv is given, save the i-th CV result only
                if ((icv > 0) && (icv != (cvset[i]))) {
                    continue;
                }

                /*
                fprintf(fp, "%d\t%g\t%g", prob->x[i].d->seqid, prob->y[i], target[i]);
                for (j=0; j<num_dec_values; j++) 
                    fprintf(fp, "\t%g", dec_values[i*num_dec_values + j]);
                fprintf(fp, "\t%d\n", cvset[i]);
                */
                fprintf(fp, "%s", prob->x[i].d->sid);
                for (j=0; j<num_dec_values; j++) 
                    fprintf(fp, "\t%g", dec_values[i*num_dec_values + j]);
                fprintf(fp, "\t%g\t%d\n", prob->y[i], cvset[i]);
            }
            if (ferror(fp) != 0 || fclose(fp) != 0)
                clog_error(CLOG(LOGGER_ID), "error occurred (%d) while writing to %s", ferror(fp), filename);
        }
    }

    clog_info(CLOG(LOGGER_ID), "Cross-Validation finished");

    free(cvset);
    free(dec_values);
    free(fold_start);
    free(perm);
}


int svm_get_svm_type(const svm_model *model)
{
    return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model)
{
    return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label)
{
    if (model->label != NULL)
        for(int i=0;i<model->nr_class;i++)
            label[i] = model->label[i];
}

void svm_get_sv_indices(const svm_model *model, int* indices)
{
    if (model->sv_indices != NULL)
        for(int i=0;i<model->l;i++)
            indices[i] = model->sv_indices[i];
}

int svm_get_nr_sv(const svm_model *model)
{
    return model->l;
}

double svm_get_svr_probability(const svm_model *model)
{
    if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
        model->probA!=NULL)
        return model->probA[0];
    else
    {
        clog_error(CLOG(LOGGER_ID), "Model doesn't contain information for SVR probability inference");
        return 0;
    }
}

double svm_predict_values(const svm_model *model, const svm_data x, double* dec_values)
{
    int i;

    int l = model->l;
    double *kvalue = Malloc(double,l);

    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
    {
        double *sv_coef = model->sv_coef[0];
        double sum = 0;

        //for speed-up
        if ((model->nr_class == 2) && 
                (model->param.kernel_type != EST_TRUNC_RBF) && 
                (model->param.kernel_type != EST_TRUNC_PW_RBF)) {
            dec_values[0] = gkmkernel_predict(x.d) - model->rho[0];

        } else { 
            gkmkernel_kernelfunc_batch_sv(x.d, kvalue);

            for(i=0;i<l;i++)
                sum += sv_coef[i] * kvalue[i];
            sum -= model->rho[0];
            *dec_values = sum;
        }

        free(kvalue);

        if(model->param.svm_type == ONE_CLASS)
            return (sum>0)?1:-1;
        else
            return sum;
    }
    else
    {
        int nr_class = model->nr_class;

        //for speed-up
        if ((nr_class == 2) && (model->param.kernel_type != EST_TRUNC_RBF) && (model->param.kernel_type != EST_TRUNC_PW_RBF)) {
            dec_values[0] = gkmkernel_predict(x.d) - model->rho[0];

            free(kvalue);

            if(dec_values[0] > 0)
                return model->label[0];
            else
                return model->label[1];
        } 

        gkmkernel_kernelfunc_batch_sv(x.d, kvalue);

        int *start = Malloc(int,nr_class);
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+model->nSV[i-1];

        int *vote = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            vote[i] = 0;


        int p=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];
                
                int k;
                double *coef1 = model->sv_coef[j-1];
                double *coef2 = model->sv_coef[i];
                for(k=0;k<ci;k++)
                    sum += coef1[si+k] * kvalue[si+k];
                for(k=0;k<cj;k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p];
                dec_values[p] = sum;

                if(dec_values[p] > 0)
                    ++vote[i];
                else
                    ++vote[j];
                p++;
            }

        int vote_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;

        free(kvalue);
        free(start);
        free(vote);
        return model->label[vote_max_idx];
    }
}

double svm_predict(const svm_model *model, const svm_data x)
{
    int nr_class = model->nr_class;
    double *dec_values;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
        dec_values = Malloc(double, 1);
    else 
        dec_values = Malloc(double, nr_class*(nr_class-1)/2);

    double pred_result = svm_predict_values(model, x, dec_values);

    free(dec_values);

    return pred_result;
}

double svm_predict_probability(
    const svm_model *model, const svm_data x, double *prob_estimates)
{
    if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
        model->probA!=NULL && model->probB!=NULL)
    {
        int i;
        int nr_class = model->nr_class;
        double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
        svm_predict_values(model, x, dec_values);

        double min_prob=1e-7;
        double **pairwise_prob=Malloc(double *,nr_class);
        for(i=0;i<nr_class;i++)
            pairwise_prob[i]=Malloc(double,nr_class);
        int k=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
                pairwise_prob[j][i]=1-pairwise_prob[i][j];
                k++;
            }
        multiclass_probability(nr_class,pairwise_prob,prob_estimates);

        int prob_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(prob_estimates[i] > prob_estimates[prob_max_idx])
                prob_max_idx = i;
        for(i=0;i<nr_class;i++)
            free(pairwise_prob[i]);
        free(dec_values);
        free(pairwise_prob);
        return model->label[prob_max_idx];
    }
    else 
        return svm_predict(model, x);
}


void svm_free_model_content(svm_model* model_ptr)
{
    if(model_ptr->sv_coef)
    {
        for(int i=0;i<model_ptr->nr_class-1;i++)
            free(model_ptr->sv_coef[i]);
    }

    free(model_ptr->SV);
    model_ptr->SV = NULL;

    free(model_ptr->sv_coef);
    model_ptr->sv_coef = NULL;

    free(model_ptr->rho);
    model_ptr->rho = NULL;

    free(model_ptr->label);
    model_ptr->label= NULL;

    free(model_ptr->probA);
    model_ptr->probA = NULL;

    free(model_ptr->probB);
    model_ptr->probB= NULL;

    free(model_ptr->sv_indices);
    model_ptr->sv_indices = NULL;

    free(model_ptr->nSV);
    model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
    if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
    {
        svm_free_model_content(*model_ptr_ptr);
        free(*model_ptr_ptr);
        *model_ptr_ptr = NULL;
    }
}

void svm_destroy_param(svm_parameter* param)
{
    free(param->weight_label);
    free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
    // svm_type

    int svm_type = param->svm_type;
    if(svm_type != C_SVC &&
       svm_type != NU_SVC &&
       svm_type != ONE_CLASS &&
       svm_type != EPSILON_SVR &&
       svm_type != NU_SVR)
        return "unknown svm type";
    
    // kernel_type
    
    int kernel_type = param->kernel_type;
    if(kernel_type != GKM &&
       kernel_type != EST_FULL &&
       kernel_type != EST_TRUNC &&
       kernel_type != EST_TRUNC_RBF &&
       kernel_type != EST_TRUNC_PW &&
       kernel_type != EST_TRUNC_PW_RBF)
        return "unknown kernel type";

    if(param->L < 2)
        return "L < 2";

    if(param->L > 12)
        return "L > 12";

    if(param->k > param->L)
        return "k > L";

    if(param->d > (param->L - param->k))
        return "d > L - k";


    // cache_size,eps,C,nu,p,shrinking

    if(param->cache_size <= 0)
        return "cache_size <= 0";

    if(param->eps <= 0)
        return "eps <= 0";

    if(svm_type == C_SVC ||
       svm_type == EPSILON_SVR ||
       svm_type == NU_SVR)
        if(param->C <= 0)
            return "C <= 0";

    if(svm_type == NU_SVC ||
       svm_type == ONE_CLASS ||
       svm_type == NU_SVR)
        if(param->nu <= 0 || param->nu > 1)
            return "nu <= 0 or nu > 1";

    if(svm_type == EPSILON_SVR)
        if(param->p < 0)
            return "p < 0";

    if(param->shrinking != 0 &&
       param->shrinking != 1)
        return "shrinking != 0 and shrinking != 1";

    if(param->probability != 0 &&
       param->probability != 1)
        return "probability != 0 and probability != 1";

    if(param->probability == 1 &&
       svm_type == ONE_CLASS)
        return "one-class SVM probability output not supported yet";


    // check whether nu-svc is feasible
    
    if(svm_type == NU_SVC)
    {
        int l = prob->l;
        int max_nr_class = 16;
        int nr_class = 0;
        int *label = Malloc(int,max_nr_class);
        int *count = Malloc(int,max_nr_class);

        int i;
        for(i=0;i<l;i++)
        {
            int this_label = (int)prob->y[i];
            int j;
            for(j=0;j<nr_class;j++)
                if(this_label == label[j])
                {
                    ++count[j];
                    break;
                }
            if(j == nr_class)
            {
                if(nr_class == max_nr_class)
                {
                    max_nr_class *= 2;
                    label = (int *)realloc(label,((size_t) max_nr_class)*sizeof(int));
                    count = (int *)realloc(count,((size_t) max_nr_class)*sizeof(int));
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }
    
        for(i=0;i<nr_class;i++)
        {
            int n1 = count[i];
            for(int j=i+1;j<nr_class;j++)
            {
                int n2 = count[j];
                if(param->nu*(n1+n2)/2 > min(n1,n2))
                {
                    free(label);
                    free(count);
                    return "specified nu is infeasible";
                }
            }
        }
        free(label);
        free(count);
    }

    return NULL;
}

int svm_check_probability_model(const svm_model *model)
{
    return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
        model->probA!=NULL && model->probB!=NULL) ||
        ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
         model->probA!=NULL);
}
