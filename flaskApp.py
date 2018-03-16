from io import BytesIO
import calendar
import flask
from flask import Flask, render_template, flash, redirect, request, url_for, session, logging, make_response,jsonify
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField,validators, BooleanField
from passlib.hash import sha256_crypt
import pygal
from pygal.style import LightStyle
import datetime

import itertools as it
from random import sample
import numpy as np
import os
import pandas as pd
import requests
import pickle
import quandl
import math
from flask import send_file
import time
from functools import wraps
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import json
import base64

#import urllib
import matplotlib.pyplot as plt
#pyplot.use('Agg')
from matplotlib import style
import gc
style.use('ggplot')

#import fix_yahoo_finance as yf
quandl.ApiConfig.api_key = 'xFzMGMyH78RLpYF6yHzs'

app = Flask(__name__)

#Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flaskApp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# init MYSQL
mysql = MySQL(app)


@app.route("/")
def main():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged in' in session:
            return f(*args, **kwargs)
        else:
            flash("You need to login first")
            return redirect(url_for('login'))
    return wrap


@app.route("/logout/")
@login_required
def logout():
     session.clear()
     flash("you have been logged out!")
     gc.collect()
     return redirect(url_for('main')) 

class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1,max=50)])
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired() ,
        validators.EqualTo('confirm', message='Passwords do not match')
     ])
    confirm = PasswordField('Confirm Password')
    accept_tos = BooleanField('I accept the <a href="/tos/"> Terms of Service</a> and the <a href="/privacy/">Privacy Notice </a>(Last updated: 27/01/2017 )', [validators.Required()])



@app.route('/register/', methods=['GET','POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        #create cursor
        cur = mysql.connection.cursor()
        x = cur.execute("SELECT * FROM users WHERE username = '%s'", (username))

        if int(x) > 0:
            flash("That username is already taken, please choose another")
        else:
            cur.execute("INSERT INTO users(name,email, username, password) VALUES(%s,%s,%s,%s)",( name, email, username, password))

        #commit to db
        mysql.connection.commit()

        #close connection
        cur.close()

        flash('You are now registered and can log in', 'success')
        gc.collect()

        session['logged in']= True
        session['username'] = username
        return redirect(url_for('main'))

    return render_template('register.html', form=form)

#user login

@app.route("/login/", methods=['GET','POST'])
def login():
    error = ''
    try:
        if request.method == "GET":
            #Get form fields
            username = request.form['username']
            password_candidate = request.form['password']

            #Create cursor
            cur = mysql.connection.cursor()

            #get user by username
            result = cur.execute("SELECT * FROM users WHERE username = '%s'" , (username))

            if result > 0:
                #GEt stored hash
                data = cur.fetchone()[2]
                password = data['password']
                
                if int(result) == 0:
                    error = "Invalid credentials, try again"
                #compare passwords
                if sha256_crypt.verify(request.form['password'], data):
                    session['logged_in'] = True
                    session['username'] = request.form['username']

                    flash("You are now logged in")
                    return redirect(url_for("dashboard"))
                else:
                    error = "Invalid login"
            gc.collect()
            
        return render_template('login.html')

    except Exception as e:
       #flash(e)
   
        return render_template('login.html')


class ReusableForm(Form):
    ticker = StringField('ticker:', validators=[validators.required()])

@app.route('/dashboard/',methods=['GET','POST'])
def dashboard():
    form = ReusableForm(request.form)
    print (form.errors)
    
    if request.method == 'POST':
        ticker = request.form['ticker']
        print(ticker)
        #df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        
        #df.set_index('date',inplace=True) 
        df = quandl.get('WIKI/{}'.format(ticker),date = { 'gte': '2016-01-01', 'lte': '2018-03-03' })
        df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    # high low -percent change(Percent volatility
        df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
        df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

        df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

        forecast_col = 'Adj. Close'
        df.fillna(-99999, inplace=True)

        forecast_out = int(math.ceil(0.01 * len(df)))
        print(forecast_out)

        df['label'] = df[forecast_col].shift(-forecast_out)


        X = np.array(df.drop(['label'], 1))
        X = preprocessing.scale(X)
        X = X[:-forecast_out]
        X_lately = X[-forecast_out:]

        df.dropna(inplace=True)
        y = np.array(df['label'])


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.2)

        clf = LinearRegression(n_jobs=-1)
        #clf= svm.SVR(kernel='poly')
        clf.fit(X_train, y_train)
        #with open('linearregression.pickle','wb') as f:
        #   pickle.dump(clf,f)

        #pickle_in = open('linearregression.pickle','rb')
        #clf = pickle.load(pickle_in)
        #clf.score(X_test,y_test)
        accuracy = clf.score(X_test, y_test)

        print(accuracy)

        forecast_set = clf.predict(X_lately)
        print(forecast_set, accuracy, forecast_out)
        df['Forecast'] = np.nan
        
        #df.set_index('date', inplace=True)
        #print(df.head())
        last_date = df.iloc[-1].name
        last_unix = last_date.timestamp()
       
        one_day = 86400
        next_unix = last_unix + one_day

        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
        
        #dataset=json.dumps(df.to_dict(orient='list'))
        
        
        df['Adj. Close'].plot()
        df['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.savefig('static/graph.png')
        plt.show()
        data = df 
    else:
        data = pd.read_csv('stock_dfs/mmm.csv')
    return render_template('dashboard.html', form=form, tables= [data.to_html()])



    
class newForm(Form):
    ticker = StringField('ticker:', validators=[validators.required()])
 

@app.route("/dataset/",methods=['GET','POST'] )
def dataset():
    form = newForm(request.form)
    print (form.errors)
    
    if request.method == "POST":
        ticker = request.form['ticker']
      
        data = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
       # data.set_index(['date'], inplace=True)
       
        print(ticker)
        
    else:
        data = pd.read_csv('stock_dfs/mmm.csv')
        
         
    return render_template('dataset.html', form=form, tables= [data.to_html()])


if __name__ == "__main__":
    app.secret_key='secret123'
    app.run(debug=True)
    #app.run()