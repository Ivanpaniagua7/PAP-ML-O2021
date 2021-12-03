"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import mse
from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

#%% Ivan

def is_nan(x):
    return (x != x)

def PredictTrain(data, p, I, q, dias):
    if p>=q:
        rezagos=p+I
    else:
        rezagos=q+I
    data=data[:-rezagos].reset_index()
    PrediccionTrain=pd.DataFrame(columns=["Predicción"])
    for i in range(rezagos,len(data)-1):
        try:
            model = SARIMAX(pd.DataFrame(data["Close"][:i]), order=(p,I,q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            diaspred=i+dias
            forecast = list(model_fit.predict(start=diaspred,end=diaspred,dynamic=True))
            PrediccionTrain.loc[data.loc[i+dias-1,"Date"]]=forecast[0] #-1 ya que aquí toma 1 dato más por el 0 del índice,
            #en predicción a 1 día, si i es 9, toma el decimo dato, es decir la fecha de la predicción.
        except:
            pass         
    return PrediccionTrain

def PredictTest(train, test, p, I, q, dias): #Este es con predicciones a diferentes dias
    if p>=q:
        rezagos=p+I
    else:
        rezagos=q+I
    data=train[-rezagos:].append(test)
    data=data.reset_index()
    PrediccionTrain=pd.DataFrame(columns=["Predicción"])
    for i in range(rezagos, len(data)):
        try:
            model = SARIMAX(pd.DataFrame(data["Close"][:i]), order=(p,I,q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            diaspred=i+dias
            forecast = list(model_fit.predict(start=diaspred,end=diaspred,dynamic=True))
            PrediccionTrain.loc[data.loc[i+dias-1,"Date"]]=forecast[0] #-1 ya que aquí toma 1 dato más por el 0 del índice,
            #en predicción a 1 día, si i es 9, toma el decimo dato, es decir la fecha del de la predicción.
        except:
            pass         
    return PrediccionTrain

def MedidasDesempeño(PredTrain, btctrain, PredTest, btctest):
    MergeTrain=pd.merge(PredTrain,btctrain, left_index=True, right_index=True)
    MergeTest=pd.merge(PredTest,btctest, left_index=True, right_index=True)
    slope, intercept, r_value, p_value, std_err = stats.linregress(MergeTrain["Predicción"],MergeTrain["Close"])
    r2train=r_value**2
    slope, intercept, r_value, p_value, std_err = stats.linregress(MergeTest["Predicción"],MergeTest["Close"])
    r2test=r_value**2
    MSETrain=mse(MergeTrain["Predicción"],MergeTrain["Close"])
    MSETest=mse(MergeTest["Predicción"],MergeTest["Close"])
    return r2train, MSETrain, r2test, MSETest

def Desempeños(Desmp1, Desmp2, Desmp3, Desmp4, dias):
    
    ModelsResults=pd.DataFrame(columns=["R\u00b2 train","MSE train", "R\u00b2 test", "MSE test"])
    ModelsResults.loc["Model 8,1,8"]=Desmp1
    ModelsResults.loc["Model 10,1,10"]=Desmp2
    ModelsResults.loc["Model 12,1,7"]=Desmp3
    ModelsResults.loc["Model 12,1,12"]=Desmp4
    
    if dias>1:
        print("\033[1m Predicciones a %s días"%dias)
    else:
        print("\033[1m Predicciones a %s día"%dias)
    
    return ModelsResults

def Estrategia(data, p, I, q, dias):
    if p>=q:
        rezagos=p+I
    else:
        rezagos=q+I
    data=data.reset_index()
    diastexto=["Actual", "1 día"]
    for i in range(2,dias+1):
        diastexto.append("%s días"%i)
    Predicciones=pd.DataFrame(columns=diastexto)
    for i in range(rezagos, len(data)):
        try:
            model = SARIMAX(pd.DataFrame(data["Close"][:i]), order=(p,I,q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            diaspred=i+dias
            forecast = list(model_fit.predict(start=i+1,end=diaspred,dynamic=True))
            Predicciones.loc[data.loc[i-1,"Date"]]=[data.loc[i-1,"Close"]]+forecast #-1 ya que aquí toma 1 dato más por el 0 del índice,
            #en predicción a 1 día, si i es 9, toma el decimo dato, es decir la fecha del de la predicción. Y estamos
            #partiendo de que la fecha que se pone es de cuando se hacen las predicciones a esos numero de días
        except:
            pass
    return Predicciones

def CrecienteDecreciente(Predicciones):
    CrecODec=Predicciones.copy()
    CrecODec["C/D"]="NA"
    CrecODec["Real 1 dia"]=CrecODec["Actual"].shift(-1)
    CrecODec["% 1 dia"]=CrecODec["Real 1 dia"]/CrecODec["Actual"]-1
    CrecODec["Real 2 dia"]=CrecODec["Actual"].shift(-2)
    CrecODec["% 2 dia"]=CrecODec["Real 2 dia"]/CrecODec["Actual"]-1
    CrecODec["Real 3 dia"]=CrecODec["Actual"].shift(-3)
    CrecODec["% 3 dia"]=CrecODec["Real 3 dia"]/CrecODec["Actual"]-1
    CrecODec["Real 4 dia"]=CrecODec["Actual"].shift(-4)
    CrecODec["% 4 dia"]=CrecODec["Real 4 dia"]/CrecODec["Actual"]-1
    CrecODec=CrecODec.dropna()
    
    COD=CrecODec.iloc
    for i in range(len(CrecODec)):
        if COD[i,1] > COD[i,0] and COD[i,2] > COD[i,1] and COD[i,3] > COD[i,2] and COD[i,4] > COD[i,3]:
            COD[i,5]="Creciente"
        elif COD[i,1] < COD[i,0] and COD[i,2] < COD[i,1] and COD[i,3] < COD[i,2] and COD[i,4] < COD[i,3]:
            COD[i,5]="Decreciente"
        else:
            COD[i,5]="Ninguno"
    return CrecODec

def DataframesCyD(PredConReales):
    Creciente=PredConReales[PredConReales["C/D"]=="Creciente"]
    Decreciente=PredConReales[PredConReales["C/D"]=="Decreciente"]
    return Creciente, Decreciente

def PromediosCambios(Creciente, Decreciente):
    Promedios=pd.DataFrame(columns=["% 1 dia","% 2 dia","% 3 dia","% 4 dia"])
    for i in range(1,5):
        c=np.mean(Creciente["% "+str(i)+" dia"])
        d=np.mean(Decreciente["% "+str(i)+" dia"])
        Promedios.loc["Creciente","% "+str(i)+" dia"]=c
        Promedios.loc["Decreciente","% "+str(i)+" dia"]=d
    return Promedios

def PreciosCoD(PreciosEstrategia):
    PreCoD=PreciosEstrategia.copy()
    PreCoD["C/D"]="NA"
    PreCoD["date"]=[PreCoD.index[i].strftime("%d/%m/%Y %H:%M") for i in range(len(PreCoD))]
    Pr=PreCoD.iloc
    for i in range(len(PreCoD)):
        if Pr[i,1] > Pr[i,0] and Pr[i,2] > Pr[i,1] and Pr[i,3] > Pr[i,2] and Pr[i,4] > Pr[i,3]:
            Pr[i,5]="Creciente"
        elif Pr[i,1] < Pr[i,0] and Pr[i,2] < Pr[i,1] and Pr[i,3] < Pr[i,2] and Pr[i,4] < Pr[i,3]:
            Pr[i,5]="Decreciente"
        else:
            Pr[i,5]="Ninguno"
    return PreCoD

def dataframes(CashInicial, inidate):
    df_operaciones=pd.DataFrame(columns=["Titulos_totales","Titulos_nuevos","Compra/venta","precio",\
                                         "Comisión","Comision_acum", "TP", "SL", "Cash_Disp",\
                                             "Cerrada", "G/P", "G/P $", "Fecha_cierre"])
    abiertas=pd.DataFrame(columns=["Titulos_totales","Titulos_nuevos","Compra/venta","precio",\
                                         "Comisión", "TP", "SL"])
    df_rentabilidad=pd.DataFrame(columns=["Valor Portafolio", "Rend Men", "Rend Acum"])
    df_rentabilidad.loc[pd.to_datetime(inidate).strftime("%d/%m/%Y %H:%M")]=CashInicial,0,0
    return df_operaciones, abiertas, df_rentabilidad

def FechasMen(precios, inidate, enddate):
    inid=pd.to_datetime(inidate)
    Fechas=[]
    dayweek=inid
    dayweek=dayweek+relativedelta(months=1)
    while dayweek<pd.to_datetime(enddate):
        Fechas.append(dayweek)
        dayweek=dayweek+relativedelta(months=1)
    Fechas=[Fechas[i].strftime("%d/%m/%Y %H:%M") for i in range(len(Fechas))]
    return Fechas

def cerrar(fechacierre, Comision, y, i, GoP):
    global CashDisponible
    #global df_operaciones
    df_operaciones.loc[y, "Cerrada"]="Si"
    #Com_Cierre=abs(df_operaciones.loc[y, "Titulos_nuevos"])*openp[i]*Comision
    if GoP=="G":
        df_operaciones.loc[y, "G/P"]="Ganadora"
        Monto_Cierre=df_operaciones.loc[y, "Titulos_nuevos"]*df_operaciones.loc[y, "TP"]
        Com_Cierre=abs(Monto_Cierre)*Comision
        df_operaciones.loc[y, "G/P $"]=abs(df_operaciones.loc[y, "TP"]-df_operaciones.loc[y, "precio"])*\
            abs(df_operaciones.loc[y, "Titulos_nuevos"])-Com_Cierre-df_operaciones.loc[y, "Comisión"]
        CashDisponible=CashDisponible+Monto_Cierre-Com_Cierre
    elif GoP=="P":
        df_operaciones.loc[y, "G/P"]="Perdedora"
        Monto_Cierre=df_operaciones.loc[y, "Titulos_nuevos"]*df_operaciones.loc[y, "SL"]
        Com_Cierre=abs(Monto_Cierre)*Comision
        df_operaciones.loc[y, "G/P $"]=-abs(df_operaciones.loc[y, "SL"]-df_operaciones.loc[y, "precio"])*\
            abs(df_operaciones.loc[y, "Titulos_nuevos"])-Com_Cierre-df_operaciones.loc[y, "Comisión"]
        CashDisponible=CashDisponible+Monto_Cierre-Com_Cierre
    else:
        df_operaciones.loc[y, "G/P"]="Indefinido"
        Monto_Cierre=df_operaciones.loc[y, "Titulos_nuevos"]*df_operaciones.loc[y, "precio"]
        Com_Cierre=abs(Monto_Cierre)*Comision
        df_operaciones.loc[y, "G/P $"]=-Com_Cierre-df_operaciones.loc[y, "Comisión"]
        CashDisponible=CashDisponible-Com_Cierre-df_operaciones.loc[y, "Comisión"]
    
    df_operaciones.loc[y, "Fecha_cierre"]=fechacierre
    abiertas.drop([y], inplace=True)
    
def Backtest(Intradia, PreciosE, CashInicial, PorcTP, PorcSL, inidate, Fechas):
    global CashDisponible
    CashDisponible=CashInicial
    time=Intradia["time"]
    closep=Intradia["close"]
    tit_tot=0
    Comision=0.00125
    Comis_acum=0
    
    global df_operaciones, abiertas, df_rentabilidad
    df_operaciones, abiertas, df_rentabilidad=dataframes(CashInicial, inidate)
    Lista=list(PreciosE["date"])
    for i in range(1,len(Intradia)):
        if Intradia["time"][i-1] in Lista:
            Tendencia=PreciosE.loc[PreciosE["date"][PreciosE["date"] == Intradia["time"][i-1]].index[0],"C/D"]
            if Tendencia!="Ninguno":
                if Tendencia=="Decreciente":
                    Volumen=0.3
                else:
                    Volumen=0.15
            #if PreciosE.loc[PreciosE["date"][PreciosE["date"] == Intradia["time"][i-1]].index[0],"C/D"]!="Ninguno":
                tit_operados=CashDisponible*Volumen/closep[i]
                tit_tot=tit_tot+tit_operados
                Comis_oper=CashDisponible*Volumen*Comision
                Comis_acum=Comis_acum+Comis_oper
                TP=closep[i]*(1+PorcTP)
                SL=closep[i]*(1-PorcSL)
                CashDisponible=CashDisponible*0.8-Comis_oper
                #df_operaciones.loc[time[i]]=tit_tot, tit_operados, "compra", closep[i],\
                df_operaciones.loc[time[i]]=tit_tot, tit_operados, Tendencia, closep[i],\
                    Comis_oper, Comis_acum, TP, SL, CashDisponible, "No","","", ""
                #abiertas.loc[time[i]]=tit_tot, tit_operados, "compra", closep[i],\
                abiertas.loc[time[i]]=tit_tot, tit_operados, Tendencia, closep[i],\
                    Comis_oper, TP, SL
                #print(PreciosE.loc[PreciosE["date"][PreciosE["date"] == Intradia["time"][i-1]].index[0],"date"],
                 #    PreciosE.loc[PreciosE["date"][PreciosE["date"] == Intradia["time"][i-1]].index[0],"C/D"])
        if len(abiertas)>0:
            for y in abiertas.index:
                #if abiertas["Compra/venta"][y]=="compra":
                if closep[i]>=abiertas["TP"][y]:
                    cerrar(time[i], Comision, y, i, "G")
                elif closep[i]<=abiertas["SL"][y]:
                    cerrar(time[i], Comision, y, i, "P")

        if time[i-1] in Fechas:
            ValorPort=CashDisponible
            if len(abiertas)>0:
                for x in abiertas.index:
                    Mont_Cierre=abiertas.loc[x, "Titulos_nuevos"]*closep[i]
                    Com_Cierr=abs(Mont_Cierre)*Comision
                    ValorPort=ValorPort+Mont_Cierre-Com_Cierr
            df_rentabilidad.loc[time[i-1], "Valor Portafolio"]=ValorPort

    for f in range(1,len(df_rentabilidad)):
        df_rentabilidad.iloc[f, 1]=df_rentabilidad.iloc[f, 0]/df_rentabilidad.iloc[f-1, 0]-1
        df_rentabilidad.iloc[f, 2]=df_rentabilidad.iloc[f, 0]/df_rentabilidad.iloc[0,0]-1

    if len(abiertas)>0:
        #print("Falta operación por cerrar")
        CashDisponible=df_rentabilidad["Valor Portafolio"][-1]
        
    return df_operaciones,abiertas, df_rentabilidad, CashDisponible





#%% Edzna

def serie_weekdays(btc):
    fecha = pd.DataFrame(btc.index)
    fecha = pd.DataFrame(pd.to_datetime(fecha["Date"]).dt.weekday)
    btc = btc.reset_index(drop=True)
    btc["Weekday"] = fecha
    days = btc.groupby("Weekday")
    df = pd.DataFrame()
    df["Lunes"] = days.get_group(0)["Close"].reset_index(drop=True)
    df["Martes"] = days.get_group(1)["Close"].reset_index(drop=True)
    df["Miércoles"] = days.get_group(2)["Close"].reset_index(drop=True)
    df["Jueves"] = days.get_group(3)["Close"].reset_index(drop=True)
    df["Viernes"] = days.get_group(4)["Close"].reset_index(drop=True)
    df["Sábado"] = days.get_group(5)["Close"].reset_index(drop=True)
    df["Domingo"] = days.get_group(6)["Close"].reset_index(drop=True)

    return df


def conteo_calculos(bitcoin):
    btc = bitcoin.copy()
    df = pd.DataFrame()
    df['CO'] = (btc['Close'] - btc['Open'])
    df['CO pips'] = df['CO'] * 0.01
    df['HO'] = (btc['High'] - btc['Open'])
    df['HO pips'] = df['HO'] * 0.01
    df['OL'] = (btc['Open'] - btc['Low'])
    df['OL pips'] = df['OL'] * 0.01
    df['HL'] = (btc['High'] - btc['Low'])
    df['HL pips'] = df['HL'] * 0.01

    return df


def estacionariedad(serie):
    plt.figure(figsize=(5, 2))
    serie.plot()
    plt.show()
    plot_acf(serie, lags=50)
    plt.show()
    plot_pacf(serie, lags=50)
    plt.show()


def pruebaDickeyFuller(btc):
    dftest = adfuller(btc)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    return dfoutput


def model_fit(btc, p, d, q):
    model = SARIMAX(btc.reset_index()["Close"], order=(p, d, q), seasonal_order=(0, 0, 0, 0))
    modelo_fit = model.fit()

    return modelo_fit


# PRUEBAS A LOS RESIDUALES
def prueba_normalidad(residuales):
    # Normalidad de los residuos Shapiro-Wilk test
    # ==============================================================================
    shapiro_test = stats.shapiro(residuales)
    print("statistic", shapiro_test[0])
    print("P-value=", shapiro_test[1])
    if shapiro_test[1] < 0.05:
        print("Los residuales se distrubuyen de forma normal.")
    else:
        print("Los residuales NO se distrubuyen de forma normal.")


def ljung_box(residuales):
    prueba = sm.stats.acorr_ljungbox(residuales, lags=[50], return_df=True)

    return prueba


def heterocedasticidad(residuales):
    prueba = breakvar_heteroskedasticity_test(residuales)

    return prueba


def Rsquared(data, modelo):
    test = data["Close"].reset_index()
    testini = test[test['Date'] == "2021-09-27"].index.values[0]
    testfin = test[test['Date'] == "2021-10-28"].index.values[0]
    test = test["Close"].loc[testini:testfin]

    forecast = modelo.predict(start=testini, end=testfin, dynamic=True)

    num = sum((test - forecast) ** 2)
    den = sum((test - test.mean()) ** 2)
    Rsq = 1 - (num / den)

    return Rsq
