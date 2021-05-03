#include <iostream>

#include <xgboost/data.h>
#include <xgboost/learner.h>
#include <xgboost/c_api.h>
#define safe_xgboost(call)                                                                             \
    {                                                                                                  \
        int err = (call);                                                                              \
        if (err != 0)                                                                                  \
        {                                                                                              \
            fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
            exit(1);                                                                                   \
        }                                                                                              \
    }

int main()
{
    // Creating toy data
    const int ROWS = 6, COLS = 2;
    const int TOTAL = ROWS * COLS;
    const float data[TOTAL] = {1, 2, 2, 4, 3, 9, 4, 8, 2, 5, 0, 1};
    DMatrixHandle dmatrix;
    safe_xgboost(XGDMatrixCreateFromMat(data, ROWS, COLS, 0, &dmatrix));

    // Loading model
    BoosterHandle model;
    XGBoosterCreate(NULL, 0, &model);
    char *model_name = "/home/viktor/Documents/plcop/results/plcop_paramodulation/policy_xgb";
    safe_xgboost(XGBoosterLoadModel(model, model_name));

    // Performing prediction
    bst_ulong output_length;
    const float *output_result;
    safe_xgboost(
        XGBoosterPredict(model, dmatrix, 0, 0, &output_length, &output_result));
    for (unsigned int i = 0; i < output_length; i++)
    {
        printf("prediction[%i] = %f \n", i, output_result[i]);
    }
}