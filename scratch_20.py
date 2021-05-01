import scipy.stats as st
import pandas as pd
import numpy as np

def get_best_distribution(data):
    # dist_names = ["alpha","anglit","arcsine","beta","betaprime","bradford","burr",
    #               "burr12","cauchy","chi","chi2","cosine", "dgamma","dweibull",
    #               "erlang","expon","exponweib","exponpow","fatiguelife","fisk",
    #               "foldcauchy","foldnorm","f","gamma","genlogistic","genpareto",
    #               "genexpon","genextreme","gengamma","genhalflogistic","geninvgauss",
    #               "gennorm","gilbrat","gompertz","gumbel_r","gumbel_l","halfcauchy",
    #               "halfnorm","halflogistic","hypsecant","gausshyper","invgamma","invgauss",
    #               "invweibull","johnsonsb","johnsonsu","ksone","kstwo","kstwobign",
    #               "laplace","laplace_asymmetric","levy_l","levy","logistic","loglaplace",
    #               "loggamma","lognorm","loguniform","maxwell","mielke","nakagami","ncx2",
    #               "ncf","nct","norm","norminvgauss","pareto","lomax","powerlognorm",
    #               "powernorm","powerlaw","rdist","rayleigh","rice","recipinvgauss","semicircular",
    #               "t","trapezoid","triang","truncexpon","truncnorm","tukeylambda","uniform",
    #               "vonmises","wald","weibull_max","weibull_min","wrapcauchy"]
    dist_names = ["alpha","anglit","arcsine","beta","betaprime","bradford","burr",
                  "burr12","cauchy","chi","chi2","cosine", "dgamma","dweibull",
                  "erlang","expon","exponweib","exponpow","fatiguelife","fisk",
                  "foldcauchy","foldnorm","f","gamma","genlogistic","genpareto",
                  "genexpon","genextreme","gengamma","genhalflogistic",
                  "gennorm","gilbrat","gompertz","gumbel_r","gumbel_l","halfcauchy",
                  "halfnorm","halflogistic","hypsecant","invgamma","invgauss",
                  "invweibull","johnsonsb","johnsonsu","kstwobign",
                  "laplace","levy_l","levy","logistic","loglaplace",
                  "loggamma","lognorm","maxwell","mielke","nakagami","ncx2",
                  "ncf","nct","norm","norminvgauss","pareto","lomax","powerlognorm",
                  "powernorm","powerlaw","rdist","rayleigh","rice","recipinvgauss","semicircular",
                  "t","triang","truncexpon","truncnorm","tukeylambda","uniform",
                  "vonmises","wald","weibull_max","weibull_min","wrapcauchy"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

data_1 = [0.00063287
,0.000919981
,0.000453382
,0.0004357
,0.00081403
,0.000554463
,0.000411904
,0.000521173
,0.000575618
,0.000676101
,0.00059141
,0.000765899
,0.000556701
,0.001002927
,0.00048721
,0.000456837
,0.000905138
,0.000404888
,0.000543683
,0.00040228
,0.000400013
,0.000468517
,0.000475484
,0.00044995
,0.000514645
,0.000406616
,0.000493894
,0.000613171
,0.000414175
,0.000408289
,0.000461937
,0.000809112
,0.000648695
,0.000462265
,0.000744978
,0.000441151
,0.000621698
,0.001107418
,0.000711287
,0.000658704
,0.001232213
,0.000438881
,0.000689811
,0.000504019
,0.000973457
,0.000605571
,0.000530908
,0.000510286
,0.001129387
,0.000669771
,0.00056926
,0.000416717
,0.000839266
,0.000422594
,0.000431284
,0.000580338
,0.000724544
,0.001061256
,0.0008886
,0.000760754
,0.000470709
,0.001014396
,0.000446521
,0.000423838
,0.000481081
,0.000781473
,0.000490052
,0.000703697
,0.001156885
,0.000532173
]
data = [0.4291
,0.2763
,0.659
,0.6977
,0.3113
,0.5003
,0.7533
,0.5382
,0.4787
,0.3953
,0.4633
,0.3348
,0.4979
,0.2527
,0.5871
,0.6501
,0.2807
,0.7775
,0.5118
,0.7875
,0.7965
,0.6217
,0.6076
,0.6674
,0.5469
,0.7711
,0.5769
,0.4444
,0.7469
,0.7654
,0.6372
,0.3133
,0.4168
,0.6364
,0.3471
,0.6868
,0.4376
,0.2261
,0.3685
,0.4091
,0.2004
,0.6914
,0.3843
,0.5619
,0.2607
,0.4506
,0.5262
,0.553
,0.2209
,0.4003
,0.4849
,0.7398
,0.302
,0.7251
,0.7066
,0.474
,0.3598
,0.2376
,0.2858
,0.3378
,0.6174
,0.2497
,0.6754
,0.7225
,0.5974
,0.3261
,0.5828
,0.3736
,0.2148
,0.5247
]

data_2 = [0.012888107
,0.016924858
,0.009360479
,0.008919462
,0.015542878
,0.011523668
,0.008320881
,0.010884417
,0.011922833
,0.013590333
,0.012205762
,0.014890193
,0.01156689
,0.017986646
,0.010185574
,0.009452434
,0.016736409
,0.008115208
,0.01131924
,0.008036855
,0.007969218
,0.009754616
,0.009919544
,0.009269987
,0.010752489
,0.008166966
,0.010326055
,0.012566972
,0.008381402
,0.008215385
,0.009586569
,0.015479581
,0.013150613
,0.009595042
,0.014590842
,0.009049719
,0.012706221
,0.019325739
,0.014118499
,0.013312688
,0.020762336
,0.008995256
,0.013802914
,0.010535423
,0.017608296
,0.012443834
,0.011074278
,0.010662975
,0.019588051
,0.013490883
,0.011806044
,0.008450183
,0.015868926
,0.008603106
,0.00881416
,0.012009861
,0.014304181
,0.018753553
,0.016519869
,0.014815986
,0.00980412
,0.018137638
,0.009181777
,0.008632445
,0.010048011
,0.015113733
,0.010244584
,0.01401253
,0.019904548
,0.011098804
]

a = 0.8507266637726878
b = 0.5536728779169697
loc = 0.00039859507831002753
scale = 0.0008606793445738634

from scipy.stats import powerlaw
from scipy.stats import johnsonsb

data_dist = pd.DataFrame(np.zeros((70,1)))
for i in range(0, 70):
    data_dist.iloc[i] = johnsonsb.ppf((i+1)/71, a,b, loc = loc, scale = scale)




a = 0.6148265313504372
b = 0.601358493417883
loc = 0.00791143540269712
scale = 0.013235019132506545

data_dist_2 = pd.DataFrame(np.zeros((70,1)))
for i in range(0, 70):
    data_dist_2.iloc[i] = johnsonsb.ppf((i+1)/71, a,b, loc = loc, scale = scale)

