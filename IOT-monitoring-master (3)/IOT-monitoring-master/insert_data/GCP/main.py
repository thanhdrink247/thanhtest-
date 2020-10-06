import datetime
import base64
import ast
import requests


def insert_data(event, context):
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    print(pubsub_message)
    res = ast.literal_eval(pubsub_message)

    if res['messagetype'] == 3:
        data = {}
        data["factory"] = res['factory_id']
        data["timestamp"] = res['timestamp']
        from datetime import datetime
        timestamp = data["timestamp"]
        date_time = datetime.fromtimestamp(timestamp + 7 * 3600)
        sdate_time = "20" + date_time.strftime("%y,%m,%d,%H,%M,%S")
        data["time"] = sdate_time
        for x, y in res['data'].items():
            data["name_machine"] = x
            data["status"] = y['status']
            data["product"] = y['product']
            data["power"] = y['power']
            insert_data_into_bigquery(data)
            push_firebase(data)
    elif res["messagetype"] == 2:
        pass
    elif res["messagetype"] == 1:
        # TODO PUSH FACTORY INFO TO FIREBASE : duong
        pass


def insert_data_into_bigquery(data):
    from datetime import datetime
    from google.cloud import bigquery
    client = bigquery.Client()
    table_id = "minus-development.dataiot.database"
    name = data["name_machine"]
    stt = data["status"]
    procduct = data["product"]
    power = data["power"]
    fact = data["factory"]
    time= datetime.fromtimestamp(data["timestamp"])
 
    timest = data["timestamp"]
    rows_to_insert = [
        {fact,name,stt,procduct,power,timest,time}
    ]

    errors = client.insert_rows_json(table_id, rows_to_insert)  # Make an API request.
    if errors == []:
        print("done")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))
        #rows = query_job.result()


def push_firebase(data):
    from datetime import datetime
    timestamp = data["timestamp"]
    date_time = datetime.fromtimestamp(timestamp)
    if (date_time.hour >= 17):
        ts = timestamp + 24 * 60 * 60
        date_time = datetime.fromtimestamp(ts)
    day = date_time.strftime("%d%m%Y")
    stt = data["status"]
    name = data["name_machine"]
    fact = data["factory"]
    number_of_products = data['product']
    timest = str(timestamp)
    import firebase_admin
    from firebase_admin import db
    app = firebase_admin.initialize_app(options={'databaseURL': 'https://minus-development.firebaseio.com'})
    path = 'factories/' + fact + '/' + name + '/' + day + '/status/' + timest
    event_ref = db.reference(path)
    event_ref.set(stt)
    path = 'factories/' + fact + '/' + name + '/' + day + '/numberOfProducts/'
    event_ref = db.reference(path)
    event_ref.set(number_of_products)
    firebase_admin.delete_app(app)

    print('done')