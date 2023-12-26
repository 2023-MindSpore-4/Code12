import mindspore
import numpy as np

def generateCVSet(X, Y, kk, index, totalCV):
    assert index <= 10
    assert totalCV <= 10
    m = X.size(0)
    slice_size = int(mindspore.ops.ceil(m / totalCV))
    start_index = (index - 1) * slice_size
    end_index = min(index * slice_size, m)

    test_x = X[kk[start_index:end_index], :]
    test_y = Y[kk[start_index:end_index], :]

    train_indices = mindspore.ops.cat((kk[:start_index], kk[end_index:]))
    train_x = X[train_indices, :]
    train_y = Y[train_indices, :]

    return train_x, train_y, test_x, test_y


def ML_TSKFS_adaptive_validate(data, target, oldOptmParameter, TSKoptions):
    optmParameter = oldOptmParameter
    alpha_searchrange = oldOptmParameter.alpha_searchrange
    beta_searchrange = oldOptmParameter.beta_searchrange
    gamma_searchrange = oldOptmParameter.gamma_searchrange

    k_searchrange = TSKoptions.k_searchrange
    h_searchrange = TSKoptions.h_searchrange
    total = (
        len(alpha_searchrange)
        * len(beta_searchrange)
        * len(gamma_searchrange)
        * len(k_searchrange)
        * len(h_searchrange)
    )
    index = 1
    parameter_cell = np.zeros((total, 35))
    ii = 1
    for p in range(len(k_searchrange)):
        for q in range(len(h_searchrange)):
            TSKoptions.k = k_searchrange[p]
            TSKoptions.h = h_searchrange[q]
            v, b = gene_ante_fcm(data, TSKoptions)
            G_data = calc_x_g(data, v, b)

            train_data = G_data
            num_train = train_data.size(0)
            randorder = mindspore.ops.randperm(num_train)

            BestResult = mindspore.ops.zeros(15, 1)
            num_cv = 5

            for i in range(len(alpha_searchrange)):
                for j in range(len(beta_searchrange)):
                    for k in range(len(gamma_searchrange)):
                        print(
                            f"\n-   {index}-th/{total}: search parameter, TSK_k = {k_searchrange[p]}, TSK_h = {h_searchrange[q]}, alpha = {alpha_searchrange[i]}, beta = {beta_searchrange[j]}, and gamma = {gamma_searchrange[k]}"
                        )
                        index += 1
                        optmParameter.alpha = alpha_searchrange[i]
                        optmParameter.beta = beta_searchrange[j]
                        optmParameter.gamma = gamma_searchrange[k]

                        optmParameter.maxIter = 100
                        optmParameter.minimumLossMargin = 0.01
                        optmParameter.outputtempresult = 0
                        optmParameter.drawConvergence = 0

                        Result = mindspore.ops.zeros(15, 1)
                        cv_index = 1
                        TempResult = mindspore.ops.zeros(num_cv, 15)

                        for cv in range(num_cv):
                            (
                                cv_train_data,
                                cv_train_target,
                                cv_test_data,
                                cv_test_target,
                            ) = generateCVSet(
                                train_data, target.t(), randorder, cv, num_cv
                            )
                            model_LLSF = ML_TSKFS(
                                cv_train_data, cv_train_target, optmParameter
                            )
                            Outputs = mindspore.ops.mm(cv_test_data, model_LLSF.t()).t()
                            Pre_Labels = mindspore.ops.round(Outputs)
                            Pre_Labels = (Pre_Labels >= 1).double()
                            TempResult[cv_index - 1, :] = EvaluationAll(
                                Pre_Labels, Outputs, cv_test_target.t()
                            ).t()
                            cv_index += 1
                        Result = mindspore.ops.mean(TempResult, dim=0).t()
                        STD = mindspore.ops.std(TempResult, dim=0)
                        if optmParameter.bQuiet == 0:
                            parameter_cell[ii - 1, 0:15] = Result[0:15, 0].t()
                            parameter_cell[ii - 1, 15:30] = STD[0, 0:15]
                            parameter_cell[ii - 1, 30] = alpha_searchrange[i]
                            parameter_cell[ii - 1, 31] = beta_searchrange[j]
                            parameter_cell[ii - 1, 32] = gamma_searchrange[k]
                            parameter_cell[ii - 1, 33] = k_searchrange[p]
                            parameter_cell[ii - 1, 34] = h_searchrange[q]
                            ii += 1
                            np.save("Parameter_cell.npy", parameter_cell)
                        r = IsBetterThanBefore(BestResult, Result)
                        if r == 1:
                            BestResult = Result
                            PrintResults(Result)
                            BestParameter = {
                                "Optm_Parameter": optmParameter,
                                "TSK_options": TSKoptions,
                            }

    return BestParameter, BestResult


def IsBetterThanBefore(Result, CurrentResult):
    a = (
        CurrentResult[1, 0]
        + CurrentResult[4, 0]
        + CurrentResult[9, 0]
        + CurrentResult[10, 0]
    )
    b = Result[1, 0] + Result[4, 0] + Result[9, 0] + Result[10, 0]

    if a > b:
        r = 1
    else:
        r = 0

    return r

