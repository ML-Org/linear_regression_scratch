import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler
import logging
from metrics import *
from utils import get_outliers
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
# for all validation purposes only
from sklearn.metrics import mean_squared_error, r2_score as R2
from scipy.stats import pearsonr


class LinearRegression():
    def __init__(self, X, y, lr=0.0001, seed=123, tolerance=1e-5, n_iter=1000, log_level=logging.INFO, type="Linear"):
        # self.data = data
        # first n as independent
        self.X = X  # data[:,:-1].reshape(-1, 1)
        # last col as target
        self.y = y.reshape(-1, 1)  # data[:,-1].reshape(-1, 1)
        self.n_cols = self.X.shape[-1] + self.y.shape[-1]
        self.lr = lr
        self.__n_iter = n_iter
        self.atol = tolerance
        self.seed = seed
        self.type_of_regression = type
        # metrics
        self.msqr = None
        self.pearson_coeff = None
        self.r2_score = None

        # configs
        np.random.seed(self.seed)
        self.logging = logging
        logging.basicConfig(level=log_level)

    def __init_theta(self):
        # one per each independent var and a bias
        return np.array([0 for i in range(self.n_cols)]).reshape(-1, 1)
        # return (np.random.sample(self.n_cols)).reshape(-1,1)
        # return np.ones(self.n_cols)

    def fit(self):
        """
        :return: returns slopes and intercept
        """
        # for bias add one column of 1's
        self._X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))
        # no theta for target variable
        self.thetas = self.__init_theta()
        stop_iter = 0
        # self.pearson_trend=[]
        # self.r2_trend=[]
        # self.msqr_trend=[]
        for _iter in range(self.__n_iter):
            old_thetas = self.thetas
            # self.logging.debug("thetas {}".format(self.thetas))
            # self.logging.debug("input : {}".format(self._X))
            # if self.type_of_regression == "linear":
            self.h_theta_X = np.matmul(self._X, self.thetas)
            # elif self.type_of_regression == "logistic":
            #     self.h_theta_X = 1/(1+np.exp(-np.matmul(self._X, self.thetas)))
            self.residuals = self.h_theta_X - self.y
            # print("sklearn msq {}".format(mean_squared_error(y_true=self.y, y_pred=h_theta_X.reshape(-1,1))))
            # self.logging.debug("residuals for thetas {} \n {} \n".format(self.thetas, self.residuals))
            self.thetas = self.update_thetas()
            stop_iter = _iter
            # if(self.atol!=0 and np.all(np.isclose(self.thetas, old_thetas, atol=self.atol))):
            #     break
            predicted_y = self.h_theta_X
            actual_y = self.y
            # self.msqr_trend.append(msqr(actual_y, predicted_y))
            # self.pearson_trend.append(pearson_corr_coef(self.h_theta_X.reshape(-1, 1), actual_y))
            # self.r2_trend.append(r2_score(actual_y, predicted_y))

        print("stop_iter {}".format(stop_iter))
        self.thetas = self.thetas.reshape(-1)
        predicted_y = self.h_theta_X
        actual_y = self.y

        # compute metrics
        self.msqr = msqr(actual_y, predicted_y)
        self.pearson_coeff = pearson_corr_coef(self.h_theta_X.reshape(-1, 1), actual_y)
        self.r2_score = r2_score(actual_y, predicted_y)
        # returns slopes and intercept
        return self.thetas[:-1], self.thetas[-1]

    def update_thetas(self):
        # print(self.residuals.shape, "   ", self._X.shape)
        # 1/m * summation(residuals*X)
        # correction  term is to be multiplied with learning rate
        # if self.type_of_regression == "linear":
        theta_correction = (np.matmul(self._X.T, self.residuals) / self._X.shape[0]) * self.lr
        # elif self.type_of_regression == "logistic":
        #     theta_correction = 1 / (1 + np.exp(-np.matmul(self._X, self.thetas)))
        self.logging.debug("theta correction \n {}".format(theta_correction))
        self.logging.debug("thetas before correction \n {}".format(self.thetas))
        thetas_corrected = self.thetas - theta_correction
        self.logging.debug("thetas after correction \n {}".format(thetas_corrected))
        return thetas_corrected

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X.dot(self.thetas.reshape(-1, 1)).reshape(-1, 1)


def preprocess_data(data, remove_outliers=True, normalize=False, seed=123):
    data = data.fillna(0)
    # if normalize:
    #     #data = (data - data.mean()) / data.std()
    #     data= pd.DataFrame(StandardScaler().fit(data).transform(data), columns = data.columns)
    if remove_outliers:
        cols = data.shape[-1]
        clean_df = pd.DataFrame()
        outliers_df = pd.DataFrame()
        for col in range(cols):
            clean_df = data[~data.iloc[:, col].isin(get_outliers(data.iloc[:, col], 2))]
            outliers_df = data[data.iloc[:, col].isin(get_outliers(data.iloc[:, col], 2))]
        logging.warning("\n {} datapoints are more than 2 std dev away from mean and removed".format(outliers_df))
    else:
        clean_df = data
    if normalize:
        clean_df = StandardScaler().fit(clean_df).transform(clean_df)
    train, test = train_test_split(np.array(clean_df), test_size=0.2, random_state=seed)

    # selecting 0 to n-1 cols
    # X_train = np.array(train[:, :-1])
    # X_test = np.array(test[:, :-1])
    # y_train = np.array(train[:, -1])
    # y_test = np.array(test[:, -1])

    X_train = np.array(train[:, :-1], dtype=np.float64)
    X_test = np.array(test[:, :-1], dtype=np.float64)
    y_train = np.array(train[:, -1]).reshape(-1, 1)
    y_test = np.array(test[:, -1]).reshape(-1, 1)
    # selecting nth columns as target column
    # y = np.array(norm_data[:, -1])
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    conf = {"normalization": True, "seed": 123}
    # X,y  = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=12, noise=100)
    data_raw = pd.read_csv("BSOM_DataSet_for_HW2.csv")

    # X_train, X_test, y_train, y_test = preprocess_data(data_raw[["all_mcqs_avg_n20","STEP_1"]], normalize=True) #"all_NBME_avg_n4"

    """
    It is obvious from the plot that there exists a good correlation as the data seems to be linear
    by normalizing data using Z scores, we verify the correlation obtained from pearson correlation coeff and the slope to be nearly same

    the R2 score is 0.54 which means the points regression line is a decent fit

    """

    X_train, X_test, y_train, y_test = preprocess_data(data_raw[["all_mcqs_avg_n20", "STEP_1"]],
                                                       normalize=conf["normalization"])

    sns.heatmap(data_raw[["all_mcqs_avg_n20", "all_NBME_avg_n4", "STEP_1"]].corr())
    plt.title("corr between two features and output")
    plt.show()

    lr = LinearRegression(X_train, y_train, lr=0.1, n_iter=1000, seed=conf["seed"], log_level=logging.INFO, tolerance=0)
    thetas, intercept = lr.fit()
    print(thetas)
    print(intercept)
    plt.scatter(X_train, y_train)
    plt.plot(X_train, X_train * thetas[0] + intercept, label="1 regressor")
    # plt.plot(X_train, X_train * 130.80991761 + 126.36649115143175, label="1 regressor")
    plt.title("Linear regression with 1 predictor with Normalization")
    plt.xlabel("all_mcqs_avg_n20")
    plt.ylabel("STEP_1")
    plt.show()

    plt.bar("pearson_train", lr.pearson_coeff, width=0.3)
    plt.bar("pearson_test", 0, width=0.3)
    plt.bar("msqr_train", lr.msqr, width=0.3)
    plt.bar("msqr_test", 0, width=0.3)
    plt.bar("r2_score_train", lr.r2_score, width=0.3)
    plt.bar("r2_score_test", 0, width=0.3)

    # plt.plot(lr.pearson_trend, label="pearson")
    # plt.plot(lr.msqr_trend, label="mean squared error")
    # plt.plot(lr.r2_trend, label="r2 score")
    # plt.xlim(0, 100)
    # plt.title("Evaluation metrics - training")
    # plt.xlabel("n_iterations")
    # plt.ylabel("scores")
    # plt.legend()
    # plt.show()

    y_pred = lr.predict(X_test)

    _msqr = msqr(y_test, y_pred)
    _pearson_coeff = pearson_corr_coef(y_pred, y_test)
    _r2_score = r2_score(y_test, y_pred)
    print(_msqr, _pearson_coeff, _r2_score, "testing")
    # plt.bar("Pearson", _pearson_coeff, width=0.3)
    # plt.bar("Pearson_2", 0, width=0.3)
    # plt.bar("msqr", _msqr, width=0.3)
    # plt.bar("msqr_2", 0, width=0.3, )
    # plt.bar("r2_score", _r2_score, width=0.3)
    # plt.bar("r2_score_2", 0, width=0.3)
    # plt.title("Evaluation metrics - testing")
    # plt.xlabel("metric")
    # plt.ylabel("scores")
    # plt.legend()
    # plt.show()

    plt.bar("pearson_train", 0, width=0.3)
    plt.bar("pearson_test", _pearson_coeff, width=0.3)
    plt.bar("msqr_train", 0, width=0.3)
    plt.bar("msqr_test", _msqr, width=0.3)
    plt.bar("r2_score_train", 0, width=0.3)
    plt.bar("r2_score_test", _r2_score, width=0.3)

    plt.title("Evaluation metrics - train vs test")
    plt.xlabel("metric")
    plt.ylabel("scores")
    plt.legend()
    plt.show()

    # thetas and intercepts
    print("thetas {} and intercept {}".format(thetas, intercept))
    # metrics
    print("Mean squared error = {}".format(lr.msqr))
    print("Pearson corr coeff = {}".format(lr.pearson_coeff))
    print("r2_score ={}".format(lr.r2_score))

    # y_pred = lr.predict(X_test)
    print("Pearson corr coeff with test data = {}".format(pearson_corr_coef(y_test.reshape(-1, 1), y_pred)))
    print("R2_score with test data {}".format(R2(y_test.reshape(-1, 1), y_pred)))
    print("Mean squared error for test data = {}".format(msqr(y_test.reshape(-1, 1), y_pred)))

    # plt.scatter(lr.h_theta_X ,lr.residuals)
    # plt.axhline(0, linestyle=":")
    # plt.xlabel("predicted values")
    # plt.ylabel("Residual (expected-predicted)")
    # plt.title("Residual vs Fit plot")
    # plt.show()

    # TODO: remove in submission
    print(""" ----------------benchmark 1 regressor start--------------- """)
    # for bench marking purposes only
    _lr = LR(fit_intercept=True).fit(X_train, y_train)
    y_pred = _lr.predict(X_test)
    print(_lr.coef_, _lr.intercept_)
    # sum of residuals are close to zero
    # print("sum of residuals {}".format(np.sum((X * _lr.coef_[0] + _lr.intercept_) - y), end="***\n"))
    print("thetas {} and intercept {} for sklearn".format(thetas, intercept))
    # plt.plot(X, X*_lr.coef_[0] + _lr.intercept_, linestyle=":")
    print("Mean squared error from sklearn = {}".format(
        mean_squared_error(X_train * _lr.coef_[0] + _lr.intercept_, y_train)))
    print("pearson r from sklearn {}".format(pearsonr(X_train * _lr.coef_[0] + _lr.intercept_, y_train.reshape(-1, 1))))
    print("pearson r from sklearn for test_data {}".format(pearsonr(y_pred.reshape(-1, 1), y_test.reshape(-1, 1))))
    print("r2 score from sklearn {}".format(R2(y_train.reshape(-1, 1), X_train * _lr.coef_[0] + _lr.intercept_)))
    print("r2 score from sklearn with test data {}".format(R2(y_pred, y_test.reshape(-1, 1))))
    print("Mean squared error from sklearn test = {}".format(mean_squared_error(y_pred, y_test)))

    print(""" ----------------benchmark 1 regressor end--------------- """)

    # disble norm
    # X_train, X_test, y_train, y_test = preprocess_data(data_raw[["all_mcqs_avg_n20", "STEP_1"]], normalize=not conf["normalization"])
    # lr = LinearRegression(X_train, y_train, lr=0.1, n_iter=1000, seed=conf["seed"], log_level=logging.INFO, tolerance=0)
    # thetas, intercept = lr.fit()
    # # print(thetas)
    # # print(intercept)
    # plt.scatter(X_train, y_train)
    # plt.plot(X_train, X_train * thetas[0] + intercept, label="1 regressor")
    # # plt.plot(X_train, X_train * 130.80991761 + 126.36649115143175, label="1 regressor")
    # plt.title("Linear regression with 1 predictor {} normalization".format("with" if conf["normalization"] else "without"))
    # plt.xlabel("all_mcqs_avg_n20")
    # plt.ylabel("STEP_1")
    # plt.show()
    #
    # plt.scatter(lr.h_theta_X, lr.residuals)
    # plt.axhline(0, linestyle=":")
    # plt.xlabel("predicted values")
    # plt.ylabel("Residual (expected-predicted)")
    # plt.title("Residual vs Fit plot for unnormalized data")
    # plt.show()

    # plt.plot(lr.pearson_trend, label="pearson")
    # plt.plot(lr.msqr_trend, label="mean squared error")
    # plt.plot(lr.r2_trend, label="r2 score")
    # plt.legend()
    # plt.show()

    # display the plot
    # plt.legend()
    # plt.show()

    # # ---------- Lr with 2 regressors-----------------#
    data_raw = pd.read_csv("BSOM_DataSet_for_HW2.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data_raw[["all_mcqs_avg_n20", "all_NBME_avg_n4", "STEP_1"]],
                                                       normalize=True, remove_outliers=True)  # "all_NBME_avg_n4"
    lr = LinearRegression(X_train, y_train, lr=0.1, seed=conf["seed"], n_iter=1000, log_level=logging.INFO, tolerance=0)
    thetas, intercept = lr.fit()

    plt.bar("pearson", 0, width=0.3)
    plt.bar("pearson_2", lr.pearson_coeff, width=0.3)
    plt.bar("msqr", 0, width=0.3)
    plt.bar("msqr_2", lr.msqr, width=0.3)
    plt.bar("r2_score", 0, width=0.3)
    plt.bar("r2_score_2", lr.r2_score, width=0.3)
    plt.xlabel("metric")
    plt.ylabel("scores")
    plt.title("Evaluation metrics - training")
    plt.show()

    y_pred = lr.predict(X_test)

    # thetas and intercepts
    print(" -- 2 regressor Training --" * 4)
    print("thetas {} and intercept {}".format(thetas, intercept))
    # metrics
    print("Mean squared error for 2 regressors = {}".format(lr.msqr))
    print("Pearson corr coeff for 2 regressors = {}".format(lr.pearson_coeff))
    print("r2_score for 2 regressors = {}".format(lr.r2_score))

    print(" -- 2 regressor Testing --" * 4)
    print("thetas {} and intercept {}".format(thetas, intercept))
    # metrics
    _msqr = msqr(y_test, y_pred)
    _pearson_coeff = pearson_corr_coef(y_pred, y_test)
    _r2_score = r2_score(y_test, y_pred)
    print("Mean squared error for 2 regressors = {}".format(_msqr))
    print("Pearson corr coeff for 2 regressors = {}".format(_pearson_coeff))
    print("r2_score for 2 regressors = {}".format(_r2_score))
    _msqr = msqr(y_test, y_pred)
    _pearson_coeff = pearson_corr_coef(y_pred, y_test)
    _r2_score = r2_score(y_test, y_pred)
    print(_msqr, _pearson_coeff, _r2_score, "testing")
    # plt.bar("Pearson", 0, width=0.3)
    # plt.bar("Pearson_2", _pearson_coeff, width=0.3)
    # plt.bar("msqr", 0, width=0.3)
    # plt.bar("msqr_2", _msqr, width=0.3, )
    # plt.bar("r2_score", 0, width=0.3)
    # plt.bar("r2_score_2", _r2_score, width=0.3)
    # plt.title("Evaluation metrics - testing")
    # plt.xlabel("metric")
    # plt.ylabel("scores")
    # plt.legend()
    # plt.show()

    # TODO: remove in submission
    print("""\n\n\n----------------benchmark 2 regressor start---------------""")
    # for bench marking purposes only
    _lr = LR(fit_intercept=True).fit(X_train, y_train)
    # coef_ = _lr.coef_.reshape[]
    coef_ = _lr.coef_.reshape(-1, 1)
    print(_lr.coef_, _lr.intercept_)
    y_pred = _lr.predict(X_test)
    # sum of residuals are close to zero
    # print("sum of residuals {}".format(np.sum((X * _lr.coef_[0] + _lr.intercept_) - y), end="***\n"))
    # plt.plot(X, X*_lr.coef_[0] + _lr.intercept_, linestyle=":")
    pred = X_train[:, 0].reshape(-1, 1) * coef_[0] + X_train[:, 1].reshape(-1, 1) * coef_[1] + _lr.intercept_
    print(" -- 2 regresor training -- " * 4)
    print("Mean squared error for 2 regressors from sklearn = {}".format(mean_squared_error(pred, y_train)))
    print("pearson r for 2 regressors from sklearn {}".format(pearsonr(pred.reshape(-1, 1), y_train.reshape(-1, 1))))
    print("r2 score for 2 regressors from sklearn {}".format(r2_score(y_train.reshape(-1, 1), pred)))
    print(" ---- ------ ---- " * 4)
    print(" -- 2 regresor testing -- " * 4)
    pred = _lr.predict(X_test)
    print("Mean squared error for 2 regressors from sklearn = {}".format(mean_squared_error(pred, y_test)))
    print("pearson r for 2 regressors from sklearn {}".format(pearsonr(pred.reshape(-1, 1), y_test.reshape(-1, 1))))
    print("r2 score for 2 regressors from sklearn {}".format(R2(y_test.reshape(-1, 1), pred)))
    print(" ---- ------ ---- " * 4)
    print("""\n\n\n----------------benchmark 2 regressor end---------------""")

    # # ---------- Lr using PCA with 1 regressor------------------#
    # """
    # It is obvious from the plot that there exists a good correlation as the data seems to be linear
    # by normalizing data using Z scores, we verify the correlation obtained from pearson correlation coeff and the slope to be nearly same
    #
    # the R2 score is 0.54 which means the points regression line is a decent fit
    #
    # """
    # data_raw = pd.read_csv("BSOM_DataSet_for_HW2.csv")
    # X_train, X_test, y_train, y_test = preprocess_data(data_raw[["all_mcqs_avg_n20", "STEP_1"]], normalize=conf["normalization"])  # "all_NBME_avg_n4"
    #
    #
    # # dimensionality reduction
    # pca = PCA()
    # pca.fit(X_train)
    # components = pca.transform(X_train)
    # print(pca.explained_variance_ratio_)
    # components_y = pca.fit_transform(y_train.reshape(-1, 1))
    # print(pca.explained_variance_ratio_)
    # X = components[:, 0].reshape(-1, 1)
    # y = components_y[:, 0].reshape(-1, 1)
    #
    #
    #
    #
    # lr = LinearRegression(X_train, y_train, lr=0.1, seed=conf["seed"], log_level=logging.INFO, tolerance=0)
    # thetas, intercept = lr.fit()
    # plt.scatter(components[:, 0], components_y[:, 0])
    # plt.plot(X, X * thetas[0] + intercept, label="1 regressor with PCA")
    # plt.show()
    # print(" -- 1 regresor PCA - training -- " * 4)
    # # thetas and intercepts
    # print("thetas {} and intercept {}".format(thetas, intercept))
    #
    # # metrics
    # print("Mean squared error = {}".format(lr.msqr))
    # print("Pearson corr coeff = {}".format(lr.pearson_coeff))
    # print("r2_score ={}".format(lr.r2_score))
    #
    # print(" ---- ------ ---- " * 4)
    #
    # print(" -- 1 regresor PCA - testing -- " * 4)
    #
    # pca = PCA()
    # pca.fit(X_test)
    # components = pca.transform(X_test)
    # components_y = pca.fit_transform(y_test.reshape(-1, 1))
    # X = components[:, 0].reshape(-1, 1)
    # pred = lr.predict(X)
    # print("Mean squared error for 2 regressors from sklearn = {}".format(mean_squared_error(pred, components_y)))
    # print("pearson r for 2 regressors from sklearn {}".format(pearsonr(pred.reshape(-1, 1), components_y.reshape(-1, 1))))
    # print("r2 score for 2 regressors from sklearn {}".format(r2_score(components_y.reshape(-1, 1), pred)))
    # print(" ---- ------ ---- " * 4)
    #
    # # # ---------- Lr using PCA with 2 regressors ------------------#
    # #
    # X_train,X_test,y_train,y_test = preprocess_data(data_raw[["all_mcqs_avg_n20", "all_NBME_avg_n4", "STEP_1"]], normalize=True)
    # # # selecting 0 to n-1 cols
    # # X = np.array(data_3_features[:, :-1])
    # # # selecting nth columns as target column
    # # y = np.array(data_3_features[:, -1])
    # pca = PCA()
    # pca.fit(X_train)
    # components = pca.transform(X_train)
    # print(pca.explained_variance_ratio_)
    # components_y = pca.fit_transform(y.reshape(-1, 1))
    # print(pca.explained_variance_ratio_)
    # X = components[:, 0].reshape(-1, 1)
    # y = components_y[:, 0].reshape(-1, 1)
    #
    # _lr_pca = LinearRegression(X, y, lr=0.1, seed=conf["seed"], log_level=logging.INFO, tolerance=0)
    # thetas, intercept = _lr_pca.fit()
    # plt.scatter(components[:,0],components_y[:,0])
    # plt.plot(X, X * thetas[0] + intercept, label="2 regressors with PCA")
    # # plt.scatter(X, y)
    # # plt.plot(X, X * thetas[0] + intercept)
    # # print((thetas, intercept))
    # print(thetas, intercept)
    # # sum of residuals are not close to zero
    # print("sum of residuals {}".format(np.sum((X * thetas[0] + intercept) - y)))
    # print(" -- 2 regresor PCA - training -- " * 4)
    # # thetas and intercepts
    # print("thetas {} and intercept {}".format(thetas, intercept))
    #
    # # metrics
    # print("Mean squared error = {}".format(_lr_pca.msqr))
    # print("Pearson corr coeff = {}".format(_lr_pca.pearson_coeff))
    # print("r2_score ={}".format(_lr_pca.r2_score))
    # print(" ---- ------ ---- " * 4)
    #
    # pca = PCA()
    # pca.fit(X_test)
    # components = pca.transform(X_test)
    # print(pca.explained_variance_ratio_)
    # components_y = pca.fit_transform(y_test.reshape(-1, 1))
    # X = components[:, 0].reshape(-1, 1)
    # print(" -- 2 regresor PCA - testing -- " * 4)
    # pred = _lr_pca.predict(X)
    # print("Mean squared error for 2 regressors = {}".format(mean_squared_error(pred, y_test)))
    # print("pearson r for 2 regressors {}".format(pearsonr(pred.reshape(-1, 1), y_test.reshape(-1, 1))))
    # print("r2 score for 2 regressors {}".format(r2_score(y_test.reshape(-1, 1), pred)))
    # print(" ---- ------ ---- " * 4)
    #
    # # display the plot
    # plt.legend()
    # plt.show()

    # data_raw = pd.read_csv("BSOM_DataSet_for_HW2.csv")
    # data=data_raw[["all_mcqs_avg_n20", "all_NBME_avg_n4", "LEVEL"]]
    # data["LEVEL"], mapping_index = pd.Series(data["LEVEL"]).factorize()
    # data = data.dropna()
    # X = np.array(data.iloc[:, :-1])
    # y = np.array(data.iloc[:, -1])
    # # selecting nth columns as targ
    # #X,y = preprocess_data(data) #"all_NBME_avg_n4"
    # print(y.shape)
    # #y = np.array(data["LEVEL"])
    # print(y.shape)
    # print(X,y)
    # print(LinearRegression(X,y, lr=0.1, tolerance=0, type="logistic").fit())
    # # print(X,y)
