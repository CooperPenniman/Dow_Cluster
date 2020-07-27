import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
style.use('ggplot')


#cluster centers


def main():
    dow = pd.read_csv('dow_jones_index.data')

   # percent return next dividend



    dow = dow.drop(["percent_change_price", "percent_change_volume_over_last_wk","previous_weeks_volume",
                    "next_weeks_open","next_weeks_close","days_to_next_dividend","percent_change_next_weeks_price","percent_return_next_dividend", "date"], axis=1)

  #  g = sns.pairplot(dow, palette= 'colorblind')

   # plt.bar(dow.stock, dow.high,  width=2, align='center')
   # plt.ylabel('high')


    #plt.scatter(dow.volume, dow.high)
  # plt.ylabel('high')
   # plt.show


    #plt.hist(dow.volume, dow.high, width=2, align='center')
    #plt.ylabel('high')
   #plt.show()



    dow['open'] = dow['open'].str.replace('$', '')
    dow['close'] = dow['close'].str.replace('$', '')
    dow['low'] = dow['low'].str.replace('$', '')
    dow['high'] = dow['high'].str.replace('$', '')

    dow['open'] = dow['open'].astype(float)
    dow['high'] = dow['high'].astype(float)
    dow['low'] = dow['low'].astype(float)
    dow['close'] = dow['close'].astype(float)

    #PCA

    dow_1 = dow.loc[:, ['close', 'low']]  # creating dataframe of fare and p class
    dow_2 = dow.loc[:, ['close', 'low']]
    pca = PCA(n_components=1)
    ins1 = pca.fit_transform(dow_1)
    ins2 = pca.fit_transform(dow_2)

    dow['proj1'] = ins1[:, 0]  # ins1 is projection on x axis
    dow['proj1'] = ins2[:, 0]


    #proj = pca.inverse_transform(dow.proj1)
    #lost = ((dow.proj1 - proj)**2).mean()
    #print(lost)

    dow_new = dow.drop(['quarter', 'open','close', 'low', 'high'], axis=1)  # dropping old columns after reinsertation
     #* why did it make me use dow_new?

   # dow_new.plot.scatter(x='close', y='low', color='blue')
  #  plt.show()

    stocktypes = {'AA'}             #makes a set of stock names without duplicates
    for stock in dow_new['stock']:
        stocktypes.add(stock)
    numclusters = len(stocktypes)

    print(dow_new.columns)

    dow_new = dow_new.drop(['stock'], axis=1) #making dow_new dataframe with only close and low


    Kmean = KMeans(n_clusters=numclusters, random_state=0)
    Kmean.fit(dow_new)
    Y_pred_1 = Kmean.predict(dow_new)
    print(dow.corr())

    plt.scatter(dow_new.volume, dow_new.proj1, c= Kmean.labels_.astype(float), s=50, alpha=.5)
    plt.scatter(Kmean.cluster_centers_[:,0], Kmean.cluster_centers_[:,1], c='red', s=50)#accessing first and second columns for the cluster
  #  plt.show() #red dots are centers of different stocks






    av_open = dow.open.mode()  # replaces empty values in fare with the mean open value
    dow['open'] = dow.open.fillna(av_open)

    av_close = dow.close.mode()  # replaces empty values in fare with the mean open value
    dow['close'] = dow.close.fillna(av_close)

    av_low = dow.low.mode()  # replaces empty values in fare with the mean open value
    dow['low'] = dow.low.fillna(av_low)

    av_high = dow.high.mode()  # replaces empty values in fare with the mean open value
    dow['high'] = dow.high.fillna(av_high)



   # dow.interpolate(method='linear', limit_direction = 'forward')



  #  print (dow.corr())











   # print(dow.columns)

if __name__ == "__main__":
    main()




