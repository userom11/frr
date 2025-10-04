import sqlite3
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

current_date = datetime.now().strftime("%Y-%m-%d")

def get_attendance_by_date(target_date):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()    
    cursor.execute(""" SELECT id, name, timestamp, date FROM attendance WHERE date = ? """, (target_date,))
    results = cursor.fetchall()
    print(f"Attendance records for {target_date}:")
    print(f"Found {len(results)} records")
    for row in results:
        print(f"ID: {row[0]}, Name: {row[1]}, Timestamp: {row[2]}, Date: {row[3]}")
    query = """ SELECT id, name, timestamp, date FROM attendance WHERE date = ? """
    df = pd.read_sql_query(query, conn, params=(target_date,))
    records = df.to_dict('records')
    json_results = { 'attendance': records }
    conn.close()
    return json_results

class MainHandler(BaseHTTPRequestHandler):  # Fixed class name
    def do_GET(self):
        # Serve index.html
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')  # Fixed content type
            self.end_headers()
            with open('index.html', 'rb') as file:
                self.wfile.write(file.read())
            return  # Important: return after handling
        
        # Handle /today
        elif self.path == '/today':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')  # Fixed content type
            self.end_headers()
            today = get_attendance_by_date(current_date)
            self.wfile.write(json.dumps(today).encode('utf-8'))  # Convert to bytes
            return
        
        # Handle /yesterday
        elif self.path == '/yesterday':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            yesterday_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday = get_attendance_by_date(yesterday_date)
            self.wfile.write(json.dumps(yesterday).encode('utf-8'))
            return
        
        # Handle date paths like /2025-10-02
        elif self.path.startswith('/') and len(self.path) == 11:  # /YYYY-MM-DD
            date_rq = self.path[1:]  # /2025-10-02 -> 2025-10-02
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            result = get_attendance_by_date(date_rq)
            self.wfile.write(json.dumps(result).encode('utf-8'))
            return
        
        # Handle 404 for unknown paths
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"404 - Page not found")

if __name__ == '__main__':
    server_address = ("127.0.0.1", 8000)
    httpd = HTTPServer(server_address, MainHandler)
    print(f"Server running on http://{server_address[0]}:{server_address[1]}")
    httpd.serve_forever()
