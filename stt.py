import pymysql.cursors
import os
import time
import schedule
import pandas as pd
import numpy as np
import datetime
 
def job(frequency, period):
   
    df = pd.read_csv('C:/Users/kaush/Desktop/ind_niftymidcap50list.csv')
    tickers = df['Symbol'].values.tolist()
 
    for ticker in tickers:
        try:
                #Make Sure Notation is Correct
                for ch in ['/','-', '+', '_', '^']:
                    if ch in ticker:
                        ticker=ticker.replace(ch,".")
                       
                x = np.array(pd.read_csv('https://www.google.com/finance/getprices?i=60&p=10d&f=d,o,h,l,c,v&df=cpct&q='+ticker,skiprows=7,header=None))
                date = []
                symbol = []
 
                #Create Date
                for i in range(0,len(x)):
                    if x[i][0][0]=='a':
                       t= datetime.datetime.fromtimestamp(int(x[i][0].replace('a',' ')))
                       date.append(t)
 
                    else:
                       date.append(t+datetime.timedelta(minutes =int(x[i][0])))
 
                #Create Symbol
                for i in range(0, len(x)):
                    symbol.append(ticker)
 
                #Create DataFrame
                df=pd.DataFrame(x)
 
                se1 = pd.Series(symbol)
                se2 = pd.Series(date)
                df['Symbol'] = se1.values
                df['Date'] = se2.values
 
                df.columns=['a', 'Open','High','Low','Close','Vol', 'Symbol', 'Date']
                df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Vol']]
                #Create Main Folder
                if not os.path.exists('C:/Users/kaush/Desktop/S&amp;P 500 Intraday Data'):
                    os.makedirs('C:/Users/kaush/Desktop/S&amp;P 500 Intraday Data')
 
                #Create Stock Folder
                ticker_path = os.path.join("C:/Users/kaush/Desktop/S&amp;P 500 Intraday Data/", ticker)
                if not os.path.exists(ticker_path):
                    os.makedirs(ticker_path)
 
                #Store CSV
                storage_path = os.path.join("C:/Users/kaush/Desktop/S&amp;P 500 Intraday Data/", ticker, ticker+".csv")
                if not os.path.exists(storage_path):
                    df.to_csv(storage_path, index=False)
                    print(ticker + ': Stored on Disk Space')
 
                #Overwrite if file already exists to save space, no need to append
                else:
                    os.remove(storage_path)
                    df.to_csv(storage_path, index=False)
                    print(ticker + ': Stored on Disk Space')
 
        except Exception as e:
            print(str(e))
def store_data():
 
        df = pd.read_csv('C:/Users/kaush/Desktop/ind_niftymidcap50list.csv')
        tickers = df['Symbol'].values.tolist()
 
        for ticker in tickers:
            read_path = os.path.join("C:/Users/kaush/Desktop/S&amp;P 500 Intraday Data/", ticker, ticker+".csv")
           
            #Open CSV
            f = open(read_path, "r")
            fstring = f.read()
 
            #Convert to List
            fList = []
            for line in fstring.split('\n'):
                fList.append(line.split(','))
               
 
            #Open Connection to Database
            connectionObject = pymysql.connect(host='localhost',
                                     user='root',
                                     port = 3306,
                                     password='',
                                     db='intraday_data')
 
            ###Create Table for Stock
            cursorObject = connectionObject.cursor()
 
            #Create Col Names from first line
            DATE = fList[0][0]; SYMBOL = fList[0][1]; OPEN = fList[0][2]; HIGH = fList[0][3]; LOW = fList[0][4]; CLOSE = fList[0][5]; VOLUME = fList[0][6]
 
 
            ###Build Table
            queryTable = """CREATE TABLE IF NOT EXISTS sp500(
                            {} DATETIME  NOT NULL,
                            {} VARCHAR(6) NOT NULL,
                            {} DECIMAL(18, 4),
                            {} DECIMAL(18, 4),
                            {} DECIMAL(18, 4),
                            {} DECIMAL(18, 4),
                            {} INT)""".format(DATE, SYMBOL, OPEN, HIGH, LOW, CLOSE, VOLUME)
 
            cursorObject.execute(queryTable)
 
 
            #Start from second row
            del fList[0]
 
            ###Generate Values
            rows = ''
            for i in range(len(fList) - 1):
                rows += "('{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(fList[i][0], fList[i][1], fList[i][2], fList[i][3], fList[i][4], fList[i][5], fList[i][6])
                if i != len(fList) - 2:
                    rows += ','
          
            #Insert
            queryInsert = "INSERT INTO sp500 VALUES" + rows
 
            try:
                cursorObject.execute(queryInsert)
                connectionObject.commit()   
            except:
                #connectionObject.rollback()
                print('Error')
            connectionObject.close()
 
            print(ticker+': Is Stored!')
 
        return store_data()

job(frequency='60',period='1')
store_data()

#schedule.every().monday.at("16:30").do(job, frequency = '60', period = '1')
#schedule.every().tuesday.at("16:30").do(job, frequency = '60', period = '1')
#schedule.every().wednesday.at("16:30").do(job, frequency = '60', period = '1')
#schedule.every().thursday.at("16:30").do(job, frequency = '60', period = '1')
#schedule.every().friday.at("16:30").do(job, frequency = '60', period = '1')
 
#while True:
   # schedule.run_pending()
    #time.sleep(60)

