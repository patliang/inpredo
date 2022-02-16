import tensorflow as tf
from keras.models import load_model
from numpy import genfromtxt
import matplotlib.pyplot as plt
import mpl_finance
import numpy as np
import uuid
import yfinance as yf
from IPython.display import Image, display

df = yf.download(tickers='qqq')
df = df.drop('Adj Close', 1)  
df['SMA_10'] = round(df['Close'].rolling(window=10).mean(), 2)
df['SMA_20'] = round(df['Close'].rolling(window=20).mean(), 2)
df['SMA_50'] = round(df['Close'].rolling(window=50).mean(), 2)
df['SMA_150'] = round(df['Close'].rolling(window=150).mean(), 2)
df['SMA_200'] = round(df['Close'].rolling(window=200).mean(), 2)
pd = ''

img_dir = '../data/test/' 
    
def predict(df): 
    df = df.tail(13) 
    #print(df)
    df.to_csv('../financial_data/qqq-test.csv', header=False)

    # Input your csv file here with historical data

    #ad = genfromtxt('../financial_data/qqq-test.csv', delimiter=',' ,dtype=str)
    #pd = np.flipud(ad)
    pd = genfromtxt('../financial_data/qqq-test.csv', delimiter=',' ,dtype=str)




    filename = graphwerk(0, 12, pd) 
    #display(Image(filename=filename))
    print(filename)
    
    
    
    #flip image
    #out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    #out 

    img_height = 150
    img_width = 150

    model_path = 'saved_model/my_model_qqq_sm10'
    model = load_model(model_path)

    #filename =  '966d56cc-f2d2-4c72-947d-9c9fe8bd5de8.jpg'

    #print('debug' + filename)
    img = tf.keras.utils.load_img('/home/patrick/notebook/inpredo/data/test/' + filename, target_size=(img_height, img_width))
    display(img)
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ['buy', 'sell']

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

#import PIL
#from PIL import Image
#read the image
#im = Image.open('/home/patrick/notebook/inpredo/data/test/' + filename)

#flip image
#out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#out 


def convolve_sma(array, period):
    return np.convolve(array, np.ones((period,))/period, mode='valid')

def graphwerk(start, finish, pd):
    open = []
    high = []
    low = []
    close = []
    volume = []
    sm10 = []
    sm20 = []
    sm50 = []
    sm150 = []
    sm200 = []
    date = []
    for x in range(finish-start+1):

# Below filtering is valid for eurusd.csv file. Other financial data files have different orders so you need to find out
# what means open, high and close in their respective order.

        open.append(float(pd[start][1]))
        high.append(float(pd[start][2]))
        low.append(float(pd[start][3]))
        close.append(float(pd[start][4]))
        volume.append(float(pd[start][5]))
        sm10.append(float(pd[start][6]))
        sm20.append(float(pd[start][7]))
        sm50.append(float(pd[start][8]))
        sm150.append(float(pd[start][9]))
        sm200.append(float(pd[start][10]))
        date.append(pd[start][0])
        start = start + 1
 
    print(sm10)
         

    fig = plt.figure(num=1, figsize=(3, 3), dpi=50, facecolor='w', edgecolor='k')
    dx = fig.add_subplot(111)
    
    # plot volume
    # create grid spec 
    #mpl_finance.volume_overlay(ax, open, close, volume, width=0.4, colorup='b', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(dx,open, close, high, low, width=1.5, colorup='g', colordown='r', alpha=0.5)

    plt.autoscale()
    plt.plot(sm10, color="orange", linewidth=10, alpha=0.5)
    plt.plot(sm20, color="purple", linewidth=10, alpha=0.5)
    plt.plot(sm50, color="gray", linewidth=10, alpha=0.5)
    plt.plot(sm150, color="yellow", linewidth=10, alpha=0.5)
    plt.plot(sm200, color="blue", linewidth=10, alpha=0.5)
    plt.axis('off') 
 
    filename = str(uuid.uuid4()) +'.jpg'
 
    plt.savefig(img_dir + filename, bbox_inches='tight') 
    plt.cla()
    plt.clf()
    return filename 


#drop last n row
for x in range(10):
    predict(df)
    df.drop(df.tail(1).index,inplace=True) # drop last n rows
    #print(df)
