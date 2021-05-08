#define PROLOG_MODULE "xgb"
#include <SWI-Prolog.h>
#include <SWI-cpp.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <math.h>
#include <iostream>

#include <xgboost/data.h>
#include <xgboost/learner.h>
#include <xgboost/c_api.h>

#define max_fea (262139 * 3 + 18)
#define safe_xgboost(call)                                                                       \
  {                                                                                              \
    int err = (call);                                                                            \
    if (err != 0)                                                                                \
    {                                                                                            \
      fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
      exit(1);                                                                                   \
    }                                                                                            \
  }

static int max_nr_attr = 64;
static unsigned int *indices;
static size_t indptr[2];
static float *data;

// these three are just to learn what works - not needed
NAMED_PREDICATE("#", hash, 2)
{
  A2 = (wchar_t *)A1;
}

PREDICATE(pi, 1)
{
  A1 = M_PI;
}

PREDICATE(write_list, 1)
{
  PlTail tail(A1);
  PlTerm e;

  while (tail.next(e))
    std::cout << (char *)e << std::endl;

  return TRUE;
}

// load the xgb predictor from a file - seems to return a funny integer
PREDICATE(predictor, 2)
{
  BoosterHandle b;
  safe_xgboost(XGBoosterCreate(NULL, 0, &b));
  safe_xgboost(XGBoosterLoadModel(b, (char *)A1));
  indices = (unsigned int *)malloc(max_nr_attr * sizeof(unsigned int));
  data = (float *)malloc(max_nr_attr * sizeof(float));
  A2 = b;
  //  A3 = indices;
  return TRUE;
}

//
// test:
// ?- load_foreign_library('xgb.so').
// ?- xgb:predictor(t,K).
// ?- time(xgb:xpredict(702196,[[0,8],[1,3],[5,8]],L)).
// % 1 inferences, 0.002 CPU in 0.002 seconds (100% CPU, 411 Lips)
// L = 0.44257044792175293.
// ?- time(dotimes(xgb:xpredict(702196,[[1,1],[3,1],[6,1]],L),1000000)).
// % 2,000,000 inferences, 4.586 CPU in 5.029 seconds (91% CPU, 436090 Lips)
// L = 0.16890858113765717 .

PREDICATE(xpredict, 3)
{
  PlTail tail(A2);
  PlTerm e, e1, e2, e3;
  int i = 0;

  // Reading prolog input into flat array
  while (tail.next(e))
  {
    if (i >= max_nr_attr - 2)
    {
      max_nr_attr *= 2;
      indices = (unsigned int *)realloc(indices, max_nr_attr * sizeof(unsigned int));
      data = (float *)realloc(data, max_nr_attr * sizeof(float));
    }
    indices[i] = (long)e[1];
    data[i] = atof((char *)e[2][1]);
    // printf("@%i: %i:%.0f\n", i, indices[i], data[i]);
    ++i;
  }
  indptr[0] = 0;
  indptr[1] = i;
  // printf("@%i: %i:%.0f\n", i, indices[i], data[i]);

  // Crteating matrix
  bst_ulong num_features;
  safe_xgboost(XGBoosterGetNumFeature(A1, &num_features));
  DMatrixHandle h_predict;
  safe_xgboost(XGDMatrixCreateFromCSREx(indptr, indices, data, 2, i, num_features, &h_predict));

  // Performing prediction
  bst_ulong out_len;
  const float *res;
  safe_xgboost(XGBoosterPredict(A1, h_predict, 0, 0, false, &out_len, &res) != 0);
  XGDMatrixFree(h_predict);
  A3 = res[0];
}
