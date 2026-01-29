# pip install flask flask-sock watchdog
import os
import time
import json
from flask import Flask, request, send_from_directory
from flask_sock import Sock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

HOST = '127.0.0.1'
PORT = 5555
WATCH_DIRECTORY = "."
WATCH_FILENAME = "watch.json"

app = Flask(__name__)
sock = Sock(app)
connections = []

def read_state():
    data = {}
    #data['_modified'] = time.time()    
    try:
        # Treat missing, empty, or only whitespace file as empty JSON object
        content = None
        if os.path.exists(WATCH_FILENAME):
            data['_modified'] = os.path.getmtime(WATCH_FILENAME)
            with open(WATCH_FILENAME, 'r') as f:
                content = f.read()
        if content and len(content.strip()) > 0:
            data = json.loads(content)
    except Exception as e:
        print(f"ERROR: Failed to read or parse {WATCH_FILENAME}: {e}")
        data['_failed'] = True
    return data

@app.route('/')
def index():
    return send_from_directory('.', 'watch.html')

@app.route('/watch.js')
def watch_js():
    return send_from_directory('.', 'watch.js')

@app.route('/watch.json')
def watch_json():
    data = read_state()
    return json.dumps(data, indent=2), 200, {'Content-Type': 'application/javascript'}

# Route any '.png' file requests
@app.route('/<path:filename>.png')
def serve_png(filename):
    if '/' in filename or '\\' in filename:
        return "Invalid filename", 400
    full_filename = f"{filename}.png"
    return send_from_directory('.', full_filename)

@sock.route('/ws')
def echo(sock):
    try:
        connections.append(sock)
        # Initial send of the current watch.json content
        data = read_state()
        try:
            data['_initial'] = True
            sock.send(json.dumps(data))
        except Exception as e:
            print(f"ERROR: Failed to send initial message to a WebSocket client: {e}")

        while True:
            data = sock.receive()
            print("WS-RECEIVE: " + data)
            #sock.send(data)
    except:
        print("ERROR: WebSocket connection error.")
    finally:
        if sock in connections:
            connections.remove(sock)

def broadcast(data):
    print('WS-BROADCAST: ' + str(data))
    message = json.dumps(data)
    # Send to all connected WebSocket clients
    for conn in connections:
        try:
            conn.send(message)
        except:
            print("ERROR: Failed to send message to a WebSocket client: " + conn)

# Watch for filesystem changes
class WatchHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            filename = event.src_path.split('/')[-1].split('\\')[-1]
            if filename == WATCH_FILENAME:
                print(f"WATCH: {event.event_type} {WATCH_FILENAME} at {event.src_path}")
                data = read_state()
                broadcast(data)

observer = Observer()
event_handler = WatchHandler()
observer.schedule(event_handler, WATCH_DIRECTORY, recursive=True)
observer.start()

# Run the Flask app
app.use_reloader=False
app.run(debug=True, host=HOST, port=PORT)

observer.stop()
observer.join()
