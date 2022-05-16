from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import statsmodels.stats.weightstats as ws


class HypothesisContainer:

    @staticmethod
    def __perform_test(df, col1, col2, alpha=0.05):
        n, _, diff, var, _, _ = stats.describe(df[col1] - df[col2])

        temp1 = df[col1].to_numpy()
        temp2 = df[col2].to_numpy()
        res = stats.ttest_rel(temp1, temp2)

        means = ws.CompareMeans(ws.DescrStatsW(temp1), ws.DescrStatsW(temp2))
        confint = means.tconfint_diff(alpha=alpha, alternative='two-sided', usevar='unequal')
        degfree = means.dof_satt()

        index = ['DegFreedom', 'Difference', 'Statistic', 'PValue', 'Low95CI', 'High95CI']
        return pd.Series([degfree, diff, res[0], res[1], confint[0], confint[1]], index=index)

    @staticmethod
    def visualize_test(df, col1, col2, alpha=0.05, title=''):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        mins = min([df[col1].min(), df[col2].min()])
        maxs = max([df[col1].max(), df[col2].max()])

        mean1 = df[col1].mean()
        mean2 = df[col2].mean()

        t_stat = HypothesisContainer.__perform_test(df, col1, col2, alpha)
        pv1 = mean2 + t_stat[4]
        pv2 = mean2 + t_stat[5]

        temp = df[col1].to_numpy()
        ax[1].hist(temp, bins=30, alpha=0.7)
        ax[1].set_xlim([mins, maxs])
        ax[1].axvline(x=mean1, color='red', linewidth=4)
        ax[1].axvline(x=pv1, color='red', linestyle='--', linewidth=4)
        ax[1].axvline(x=pv2, color='red', linestyle='--', linewidth=4)
        ax[1].set_ylabel('Count')
        ax[1].set_xlabel(col1)

        temp = df[col2].to_numpy()
        ax[0].hist(temp, bins=30, alpha=0.7)
        ax[0].set_xlim([mins, maxs])
        ax[0].axvline(x=mean2, color='red', linewidth=4)
        ax[0].set_ylabel('Count')
        ax[0].set_xlabel(col2)

        plt.suptitle(title)

        return t_stat
