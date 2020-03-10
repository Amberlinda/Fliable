from django.shortcuts import render
import requests, re
from fliable_app.predictions.arima_model import AULVJ
import django.http

def home(request):
    return render(request,'index.html',{})
    
def result(request):
    
    url = "http://developer.goibibo.com/api/search/?app_id=466724b6&app_key=d57c25f7b56dcc85f6af087663e93454"
    
    PARAMS = {
        # 'app_id':466724,
        # 'app_key':"d57c25f7b56dcc85f6af087663e93454",
        'format':"json",
        'source':"BLR",
        'destination':"DEL",
        'dateofdeparture':20200331,
        'class':"E",
        'adults':1,
        'counter':100
    }
    
    if request.method == 'POST':
        #getting date from the form
        date = request.POST.get('date')
        clean_date = re.sub(r'[^\w]', '', date)
        #modifying parameters of the api
        PARAMS['dateofdeparture'] = clean_date
        #sending request to api
        response = requests.get(url = url,params=PARAMS)
        data = response.json()
        #slicing the json data
        text = data["data"]["onwardflights"]
        length = len(data["data"]["onwardflights"])
        return render(request,'result.html',{'data':text,'length':length,'date':date})
    else:
        response = requests.get(url = url,params=PARAMS)
        data = response.json()
        text = data["data"]["onwardflights"]
        length = len(data["data"]["onwardflights"])
        return render(request,'result.html',{'data':text,'length':length})
    
    
def predictions(request):

    if request.method == 'POST':
        airline_id = request.POST.get("airline_id")
        predictions  = AULVJ(airline_id)
        if predictions:
            return render(request,'prediction.html',{'predict':predictions[0],'airline':airline_id,'future_prices':predictions[1]})
    else:
        return render(request,'prediction.html',{})
       
    
    